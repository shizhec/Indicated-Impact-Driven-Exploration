from slurm_processor import load_slurm_output
import matplotlib.pyplot as plt
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