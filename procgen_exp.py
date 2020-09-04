from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_normalized_mean_return(Rmax, Rmin, path, filename):
    output_dict = load_slurm_output(path, filename)

    frames = np.array(list(output_dict.keys()))
    mean_episode_return = np.array([x.get("mean_episode_return", 0.0) for x in output_dict.values()])

    normalized_mean_return = (mean_episode_return - Rmin) / (Rmax - Rmin)

    plt.plot(frames, mean_episode_return)
    plt.title("frames vs mean_episode_return")
    plt.show()

    plt.plot(frames, normalized_mean_return)
    plt.title("frames vs normalized mean return")

    plt.show()

def tuning_learning_rate(path, game):
    output1 = load_slurm_output(path, game + "_rl_001.out")
    output2 = load_slurm_output(path, game + "_rl_005.out")
    output3 = load_slurm_output(path, game + "_rl_0001.out")
    output4 = load_slurm_output(path, game + "_rl_0005.out")
    output5 = load_slurm_output(path, game + "_rl_00005.out")

    mean_episode_return1 = np.array([x.get("mean_episode_return", 0.0) for x in output1.values()])
    mean_episode_return2 = np.array([x.get("mean_episode_return", 0.0) for x in output2.values()])
    mean_episode_return3 = np.array([x.get("mean_episode_return", 0.0) for x in output3.values()])
    mean_episode_return4 = np.array([x.get("mean_episode_return", 0.0) for x in output4.values()])
    mean_episode_return5 = np.array([x.get("mean_episode_return", 0.0) for x in output5.values()])

    print("mean_return_rl_0.001: ", np.mean(mean_episode_return1))
    print("mean_return_rl_0.005: ", np.mean(mean_episode_return2))
    print("mean_return_rl_0.0001: ", np.mean(mean_episode_return3))
    print("mean_return_rl_0.0005: ", np.mean(mean_episode_return4))
    print("mean_return_rl_0.00005: ", np.mean(mean_episode_return5))

def tuning_batch_size(path, game):
    pass

def load_slurm_output(path, filename):
    output_dict = OrderedDict()
    with open(path+filename, "r") as file:
        for line in file:
            if line.startswith('[') and "frames" in line:
                frames = [int(s) for s in line.split() if s.isdigit()][0]
            else:
                line = line.strip("\n")
                line = line.replace("nan", "0.0")
                if line.startswith('{'):
                    lines = line
                    if '}' in line:
                        stat_dict = eval(lines)
                        output_dict[frames] = stat_dict
                elif line.endswith("}"):
                    lines += line
                    stat_dict = eval(lines)
                    output_dict[frames] = stat_dict
                else:
                    lines += line

    return output_dict


# plot_normalized_mean_return(10, 5, "coinrun")
print("caveflyer:")
tuning_learning_rate("data/caveflyer/", "caveflyer")
print("climber:")
tuning_learning_rate("data/climber/", "climber")
print("coinrun:")
tuning_learning_rate("data/coinrun/", "coinrun")
print("jumper:")
tuning_learning_rate("data/jumper/", "jumper")
print("leaper:")
tuning_learning_rate("data/leaper/", "leaper")
print("ninja")
tuning_learning_rate("data/ninja/", "ninja")