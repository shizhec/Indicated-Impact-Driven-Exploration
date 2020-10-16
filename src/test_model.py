import torch
import os
from src.utils import create_env
from src.env_utils import ProcGenEnvironment
import pickle
from src import models
import numpy as np
from datetime import datetime

ProcGenStateEmbeddingNet = models.ProcGenStateEmbeddingNet
ProcGenForwardDynamicsNet = models.ProcGenForwardDynamicsNet
ProcGenInverseDynamicsNet = models.ProcGenInverseDynamicsNet
ProcGenPolicyNet = models.ProcGenPolicyNet


def test(flags):
    checkpointpath = os.path.expandvars(os.path.expanduser(
        '%s%s%s/' % (flags.savedir, 'data/'+flags.game+'/'+flags.model+'/generalization/'+flags.model+'-',
                       flags.modelpath)))
    print(checkpointpath)

    checkpoint = torch.load(checkpointpath+'model.tar')

    env = create_env(flags)
    model = ProcGenPolicyNet(env.observation_space.shape, env.action_space.n, flags)
    model.load_state_dict(checkpoint['model_state_dict'])
    initial_obs = env.reset()
    initial_batch = batch_frame(initial_obs)

    model_output, _ = model(initial_batch)
    action = model_output['action']

    current_step = 0
    episode_reward = []
    current_episode_reward = 0
    print("start time: ", datetime.now(), flush=True)
    while current_step < flags.total_frames:
        if current_step % 100000 == 0:
            print(current_step, " ", np.mean(episode_reward), flush=True)
        obs, reward, done, _ = env.step(action)

        current_episode_reward += reward
        if done:
            obs = env.reset()
            episode_reward.append(current_episode_reward)
            current_episode_reward = 0
        obs_batch = batch_frame(obs)

        model_output, _ = model(obs_batch)
        action = model_output['action']

        current_step += 1

    outputpath = checkpointpath+flags.test_out_file + ".pkl"
    print(outputpath)
    with open(outputpath, "wb") as fp:
        pickle.dump(episode_reward, fp)

    print("end time: ", datetime.now(), flush=True)

    # with open("test_output.pkl", "rb") as fp:
    #     rewards = pickle.load(fp)
    #     print(np.sum(np.array(rewards)))

def batch_frame(observation):
    batch = torch.tensor([[observation]])
    return batch
