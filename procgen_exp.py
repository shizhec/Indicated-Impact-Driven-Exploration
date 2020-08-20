from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


def plot_normalized_mean_return(Rmax, Rmin, env):
    output_dict = load_slurm_output(env)

    frames = np.array(list(output_dict.keys()))
    mean_episode_return = np.array([x.get("mean_episode_return", 0.0) for x in output_dict.values()])

    normalized_mean_return = (mean_episode_return - Rmin) / (Rmax - Rmin)

    plt.plot(frames, mean_episode_return)
    plt.title("frames vs mean_episode_return")
    plt.show()

    plt.plot(frames, normalized_mean_return)
    plt.title("frames vs normalized mean return")

    plt.show()



def load_slurm_output(env_name):
    output_dict = OrderedDict()
    with open("data/"+env_name+".out", "r") as file:
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


plot_normalized_mean_return(10, 5, "coinrun")