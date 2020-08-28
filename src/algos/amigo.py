# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Must be run with OMP_NUM_THREADS=1

import logging
import os
import threading
import time
import timeit
import pprint

import numpy as np
from torch.distributions.normal import Normal
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

from src.env_utils import FrameStack
from src.utils import get_batch, log, create_env, create_buffers_amigo, act_amigo

ProcGenPolicyNet = models.ProcGenPolicyNet
Generator = models.Generator

# Some Global Variables
# We start t* at 7 steps.
generator_batch = dict()
generator_batch_aux = dict()
generator_current_target = 7.0
generator_count = 0

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

def reached_goal_func(frames, goals, initial_frames = None, done_aux = None):
    """Auxiliary function which evaluates whether agent has reached the goal."""
    new_frame = torch.flatten(frames, 2, 3)
    old_frame = torch.flatten(initial_frames, 2, 3)
    ans = new_frame == old_frame
    ans = torch.sum(ans, 3) != 3  # reached if the three elements are not the same
    reached = torch.squeeze(torch.gather(ans, 2, torch.unsqueeze(goals.long(),2)))
    return reached

def learn(
        actor_model,
        model,
        actor_generator_model,
        generator_model,
        batch,
        initial_agent_state,
        optimizer,
        generator_model_optimizer,
        scheduler,
        generator_scheduler,
        flags,
        max_steps=100.0,
        lock=threading.Lock()
):
    """Performs a learning (optimization) step for the policy, and for the generator whenever the generator batch is full."""
    with lock:

        # Loading Batch
        next_frame = batch['frame'][1:].float().to(device=flags.device)
        initial_frames = batch['initial_frame'][1:].float().to(device=flags.device)
        done_aux = batch['done'][1:].float().to(device=flags.device)
        reached_goal = reached_goal_func(next_frame, batch['goal'][1:].to(device=flags.device),
                                         initial_frames=initial_frames, done_aux=done_aux)
        intrinsic_rewards = flags.intrinsic_reward_coef * reached_goal.float()
        reached = reached_goal.type(torch.bool)
        intrinsic_rewards = intrinsic_rewards * (
                    intrinsic_rewards - 0.9 * (batch["episode_step"][1:].float() / max_steps))

        learner_outputs, unused_state = model(batch, initial_agent_state)
        bootstrap_value = learner_outputs["baseline"][-1]
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}
        rewards = batch["reward"]

        # Student Rewards
        if flags.no_generator:
            total_rewards = rewards
        elif flags.no_extrinsic_rewards:
            total_rewards = intrinsic_rewards
        else:
            total_rewards = rewards + intrinsic_rewards

        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(total_rewards, -1, 1)
        elif flags.reward_clipping == "soft_asymmetric":
            squeezed = torch.tanh(total_rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = torch.where(total_rewards < 0, 0.3 * squeezed, squeezed) * 5.0
        elif flags.reward_clipping == "none":
            clipped_rewards = total_rewards
        discounts = (~batch["done"]).float() * flags.discounting
        clipped_rewards += 1.0 * (rewards > 0.0).float()

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        # Student Loss
        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = losses.compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]

        if torch.isnan(torch.mean(episode_returns)):
            aux_mean_episode = 0.0
        else:
            aux_mean_episode = torch.mean(episode_returns).item()
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": aux_mean_episode,
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "gen_rewards": None,
            "gg_loss": None,
            "generator_baseline_loss": None,
            "generator_entropy_loss": None,
            "mean_intrinsic_rewards": None,
            "mean_episode_steps": None,
            "ex_reward": None,
            "generator_current_target": None,
        }

        if flags.no_generator:
            stats["gen_rewards"] = 0.0,
            stats["gg_loss"] = 0.0,
            stats["generator_baseline_loss"] = 0.0,
            stats["generator_entropy_loss"] = 0.0,
            stats["mean_intrinsic_rewards"] = 0.0,
            stats["mean_episode_steps"] = 0.0,
            stats["ex_reward"] = 0.0,
            stats["generator_current_target"] = 0.0,

        scheduler.step()
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()
        actor_model.load_state_dict(model.state_dict())

        # Generator:
        if not flags.no_generator:
            global generator_batch
            global generator_batch_aux
            global generator_current_target
            global generator_count
            global goal_count_dict

            # Loading Batch
            is_done = batch['done'] == 1
            reached = reached_goal.type(torch.bool)
            if 'frame' in generator_batch.keys():
                generator_batch['frame'] = torch.cat(
                    (generator_batch['frame'], batch['initial_frame'][is_done].float().to(device=flags.device)), 0)
                generator_batch['goal'] = torch.cat(
                    (generator_batch['goal'], batch['goal'][is_done].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat(
                    (generator_batch['episode_step'], batch['episode_step'][is_done].float().to(device=flags.device)),
                    0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'],
                                                                 batch['generator_logits'][is_done].float().to(
                                                                     device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'],
                                                        torch.zeros(batch['goal'].shape)[is_done].float().to(
                                                            device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat(
                    (generator_batch['ex_reward'], batch['reward'][is_done].float().to(device=flags.device)), 0)
                generator_batch['carried_obj'] = torch.cat(
                    (generator_batch['carried_obj'], batch['carried_obj'][is_done].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat(
                    (generator_batch['carried_col'], batch['carried_col'][is_done].float().to(device=flags.device)), 0)

                generator_batch['carried_obj'] = torch.cat(
                    (generator_batch['carried_obj'], batch['carried_obj'][reached].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat(
                    (generator_batch['carried_col'], batch['carried_col'][reached].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat(
                    (generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0)
                generator_batch['frame'] = torch.cat(
                    (generator_batch['frame'], batch['initial_frame'][reached].float().to(device=flags.device)), 0)
                generator_batch['goal'] = torch.cat(
                    (generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat(
                    (generator_batch['episode_step'], batch['episode_step'][reached].float().to(device=flags.device)),
                    0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'],
                                                                 batch['generator_logits'][reached].float().to(
                                                                     device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'],
                                                        torch.ones(batch['goal'].shape)[reached].float().to(
                                                            device=flags.device)), 0)
            else:
                generator_batch['frame'] = (batch['initial_frame'][is_done]).float().to(
                    device=flags.device)  # Notice we use initial_frame from batch
                generator_batch['goal'] = (batch['goal'][is_done]).to(device=flags.device)
                generator_batch['episode_step'] = (batch['episode_step'][is_done]).float().to(device=flags.device)
                generator_batch['generator_logits'] = (batch['generator_logits'][is_done]).float().to(
                    device=flags.device)
                generator_batch['reached'] = (torch.zeros(batch['goal'].shape)[is_done]).float().to(device=flags.device)
                generator_batch['ex_reward'] = (batch['reward'][is_done]).float().to(device=flags.device)
                generator_batch['carried_obj'] = (batch['carried_obj'][is_done]).float().to(device=flags.device)
                generator_batch['carried_col'] = (batch['carried_col'][is_done]).float().to(device=flags.device)

                generator_batch['carried_obj'] = torch.cat(
                    (generator_batch['carried_obj'], batch['carried_obj'][reached].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat(
                    (generator_batch['carried_col'], batch['carried_col'][reached].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat(
                    (generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0)
                generator_batch['frame'] = torch.cat(
                    (generator_batch['frame'], batch['initial_frame'][reached].float().to(device=flags.device)), 0)
                generator_batch['goal'] = torch.cat(
                    (generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat(
                    (generator_batch['episode_step'], batch['episode_step'][reached].float().to(device=flags.device)),
                    0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'],
                                                                 batch['generator_logits'][reached].float().to(
                                                                     device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'],
                                                        torch.ones(batch['goal'].shape)[reached].float().to(
                                                            device=flags.device)), 0)

            if generator_batch['frame'].shape[
                0] >= flags.generator_batch_size:  # Run Gradient step, keep batch residual in batch_aux
                for key in generator_batch:
                    generator_batch_aux[key] = generator_batch[key][flags.generator_batch_size:]
                    generator_batch[key] = generator_batch[key][:flags.generator_batch_size].unsqueeze(0)

                generator_outputs = generator_model(generator_batch)
                generator_bootstrap_value = generator_outputs["generator_baseline"][-1]

                # Generator Reward
                def distance2(episode_step, reached, targ=flags.generator_target):
                    aux = flags.generator_reward_negative * torch.ones(episode_step.shape).to(device=flags.device)
                    aux += (episode_step >= targ).float() * reached
                    return aux

                if flags.generator_loss_form == 'gaussian':
                    generator_target = flags.generator_target * torch.ones(generator_batch['episode_step'].shape).to(
                        device=flags.device)
                    gen_reward = Normal(generator_target, flags.target_variance * torch.ones(generator_target.shape).to(
                        device=flags.device))
                    generator_rewards = flags.generator_reward_coef * (
                                2 + gen_reward.log_prob(generator_batch['episode_step']) - gen_reward.log_prob(
                            generator_target)) * generator_batch['reached'] - 1

                elif flags.generator_loss_form == 'linear':
                    generator_rewards = (generator_batch['episode_step'] / flags.generator_target * (
                                generator_batch['episode_step'] <= flags.generator_target).float() + \
                                         torch.exp(
                                             (-generator_batch['episode_step'] + flags.generator_target) / 20.0) * (
                                                     generator_batch[
                                                         'episode_step'] > flags.generator_target).float()) * \
                                        2 * generator_batch['reached'] - 1


                elif flags.generator_loss_form == 'dummy':
                    generator_rewards = torch.tensor(
                        distance2(generator_batch['episode_step'], generator_batch['reached'])).to(device=flags.device)

                elif flags.generator_loss_form == 'threshold':
                    generator_rewards = torch.tensor(
                        distance2(generator_batch['episode_step'], generator_batch['reached'],
                                  targ=generator_current_target)).to(device=flags.device)

                if torch.mean(generator_rewards).item() >= flags.generator_threshold:
                    generator_count += 1
                else:
                    generator_count = 0

                if generator_count >= flags.generator_counts and generator_current_target <= flags.generator_maximum:
                    generator_current_target += 1.0
                    generator_count = 0
                    goal_count_dict *= 0.0

                if flags.novelty:
                    frames_aux = torch.flatten(generator_batch['frame'], 2, 3)
                    frames_aux = frames_aux[:, :, :, 0]
                    object_ids = torch.zeros(generator_batch['goal'].shape).long()
                    for i in range(object_ids.shape[1]):
                        object_ids[0, i] = frames_aux[0, i, generator_batch['goal'][0, i]]
                        goal_count_dict[object_ids[0, i]] += 1

                    bonus = (object_ids > 2).float().to(device=flags.device) * flags.novelty_bonus
                    generator_rewards += bonus

                if flags.reward_clipping == "abs_one":
                    generator_clipped_rewards = torch.clamp(generator_rewards, -1, 1)

                if not flags.no_extrinsic_rewards:
                    generator_clipped_rewards = 1.0 * (
                                generator_batch['ex_reward'] > 0).float() + generator_clipped_rewards * (
                                                            generator_batch['ex_reward'] <= 0).float()

                generator_discounts = torch.zeros(generator_batch['episode_step'].shape).float().to(device=flags.device)

                goals_aux = generator_batch["goal"]
                if flags.inner:
                    goals_aux = goals_aux.float()
                    goals_aux -= 2 * (torch.floor(goals_aux / generator_model.height))
                    goals_aux -= generator_model.height - 1
                    goals_aux = goals_aux.long()

                generator_vtrace_returns = vtrace.from_logits(
                    behavior_policy_logits=generator_batch["generator_logits"],
                    target_policy_logits=generator_outputs["generator_logits"],
                    actions=goals_aux,
                    discounts=generator_discounts,
                    rewards=generator_clipped_rewards,
                    values=generator_outputs["generator_baseline"],
                    bootstrap_value=generator_bootstrap_value,
                )

                # Generator Loss
                gg_loss = losses.compute_policy_gradient_loss(
                    generator_outputs["generator_logits"],
                    goals_aux,
                    generator_vtrace_returns.pg_advantages,
                )

                generator_baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
                    generator_vtrace_returns.vs - generator_outputs["generator_baseline"]
                )

                generator_entropy_loss = flags.generator_entropy_cost * losses.compute_entropy_loss(
                    generator_outputs["generator_logits"]
                )

                generator_total_loss = gg_loss + generator_entropy_loss + generator_baseline_loss

                intrinsic_rewards_gen = generator_batch['reached'] * (
                            1 - 0.9 * (generator_batch["episode_step"].float() / max_steps))
                stats["gen_rewards"] = torch.mean(generator_clipped_rewards).item()
                stats["gg_loss"] = gg_loss.item()
                stats["generator_baseline_loss"] = generator_baseline_loss.item()
                stats["generator_entropy_loss"] = generator_entropy_loss.item()
                stats["mean_intrinsic_rewards"] = torch.mean(intrinsic_rewards_gen).item()
                stats["mean_episode_steps"] = torch.mean(generator_batch["episode_step"]).item()
                stats["ex_reward"] = torch.mean(generator_batch['ex_reward']).item()
                stats["generator_current_target"] = generator_current_target

                generator_scheduler.step()
                generator_model_optimizer.zero_grad()
                generator_total_loss.backward()

                nn.utils.clip_grad_norm_(generator_model.parameters(), 40.0)
                generator_model_optimizer.step()
                actor_generator_model.load_state_dict(generator_model.state_dict())

                if generator_batch_aux['frame'].shape[0] > 0:
                    generator_batch = {key: tensor[:] for key, tensor in generator_batch_aux.items()}
                else:
                    generator_batch = dict()

        return stats


def train(flags):
    """Full training loop."""
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)

    # env = wrappers.FullyObsWrapper(env)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)

    generator_model = Generator(env.observation_space.shape, env.observation_space.shape[0],
                                       env.observation_space.shape[1],
                                num_input_frames=flags.num_input_frames)

    model = ProcGenPolicyNet(env.observation_space.shape, env.action_space.n)

    global goal_count_dict
    goal_count_dict = torch.zeros(11).float().to(device=flags.device)

    logits_size = env.observation_space.shape[0] * env.observation_space.shape[1]

    buffers = create_buffers_amigo(env.observation_space.shape, model.num_actions, flags, logits_size)

    model.share_memory()
    generator_model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act_amigo,
            args=(i, free_queue, full_queue, model, generator_model, buffers,
                 initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    learner_model = ProcGenPolicyNet(env.observation_space.shape, env.action_space.n) \
        .to(device=flags.device)
    learner_generator_model = Generator(env.observation_space.shape, env.observation_space.shape[0],
                                               env.observation_space.shape[1],
                                               num_input_frames=flags.num_input_frames).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    generator_model_optimizer = torch.optim.RMSprop(
        learner_generator_model.parameters(),
        lr=flags.generator_learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_model_optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "gen_rewards",
        "gg_loss",
        "generator_entropy_loss",
        "generator_baseline_loss",
        "mean_intrinsic_rewards",
        "mean_episode_steps",
        "ex_reward",
        "generator_current_target",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state = get_batch(free_queue, full_queue, buffers,
                initial_agent_state_buffers, flags, timings)
            stats = learn(model, learner_model, generator_model, learner_generator_model,
                          batch, agent_state, optimizer, generator_model_optimizer, scheduler,
                          generator_scheduler, flags)

            timings.time("learn")
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "generator_model_state_dict": generator_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_model_optimizer_state_dict": generator_model_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "generator_scheduler_state_dict": generator_scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()

        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)
            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                        "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "After %i frames: loss %f @ %.1f fps. %sStats:\n%s",
                frames,
                total_loss,
                fps,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d frames.", frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()

