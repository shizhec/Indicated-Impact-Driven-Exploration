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
    output1 = load_slurm_output(path, game + "_bs_8.out", 51)
    output2 = load_slurm_output(path, game + "_bs_32.out", 51)

    mean_episode_return1 = np.array([x.get("mean_episode_return", 0.0) for x in output1.values()])
    mean_episode_return2 = np.array([x.get("mean_episode_return", 0.0) for x in output2.values()])

    print("mean_return_bs_8: ", np.mean(mean_episode_return1))
    print("mean_return_bs_32: ", np.mean(mean_episode_return2))

def tuning_unroll_length(path, game):
    output1 = load_slurm_output(path, game + "_ul_20.out", 51)
    output2 = load_slurm_output(path, game + "_ul_40.out", 51)
    output3 = load_slurm_output(path, game + "_ul_60.out", 51)
    output4 = load_slurm_output(path, game + "_ul_80.out", 51)
    output5 = load_slurm_output(path, game + "_ul_100.out", 51)

    mean_episode_return1 = np.array([x.get("mean_episode_return", 0.0) for x in output1.values()])
    mean_episode_return2 = np.array([x.get("mean_episode_return", 0.0) for x in output2.values()])
    mean_episode_return3 = np.array([x.get("mean_episode_return", 0.0) for x in output3.values()])
    mean_episode_return4 = np.array([x.get("mean_episode_return", 0.0) for x in output4.values()])
    mean_episode_return5 = np.array([x.get("mean_episode_return", 0.0) for x in output5.values()])

    print("mean_return_ul_20: ", np.mean(mean_episode_return1))
    print("mean_return_ul_40: ", np.mean(mean_episode_return2))
    print("mean_return_ul_60: ", np.mean(mean_episode_return3))
    print("mean_return_ul_80: ", np.mean(mean_episode_return4))
    print("mean_return_ul_100: ", np.mean(mean_episode_return5))

def tuning_irc(path, game):
    output1 = [load_slurm_output(path, game + "_irc_10_"+str(i)+".out", 51) for i in range(1,6)]
    output2 = [load_slurm_output(path, game + "_irc_1_"+str(i)+".out", 51) for i in range(1,6)]
    output3 = [load_slurm_output(path, game + "_irc_5_"+str(i)+".out", 51) for i in range(1,6)]
    output4 = [load_slurm_output(path, game + "_irc_01_"+str(i)+".out", 51) for i in range(1,6)]
    output5 = [load_slurm_output(path, game + "_irc_05_"+str(i)+".out", 51) for i in range(1,6)]
    output6 = [load_slurm_output(path, game + "_irc_001_"+str(i)+".out", 51) for i in range(1,6)]
    output7 = [load_slurm_output(path, game + "_irc_005_"+str(i)+".out", 51) for i in range(1,6)]


    mean_episode_return1 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output1]
    mean_episode_return2 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output2]
    mean_episode_return3 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output3]
    mean_episode_return4 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output4]
    mean_episode_return5 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output5]
    mean_episode_return6 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output6]
    mean_episode_return7 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output7]

    print("mean_return_irc_10: ", np.mean(mean_episode_return1))
    print("mean_return_irc_1: ", np.mean(mean_episode_return2))
    print("mean_return_irc_5: ", np.mean(mean_episode_return3))
    print("mean_return_irc_01: ", np.mean(mean_episode_return4))
    print("mean_return_irc_05: ", np.mean(mean_episode_return5))
    print("mean_return_irc_001: ", np.mean(mean_episode_return6))
    print("mean_return_irc_005: ", np.mean(mean_episode_return7))

def tuning_ec(path, game):
    output1 = [load_slurm_output(path, game + "_ec_01_"+str(i)+".out", 51) for i in range(1,6)]
    output2 = [load_slurm_output(path, game + "_ec_001_"+str(i)+".out", 51) for i in range(1,6)]
    output3 = [load_slurm_output(path, game + "_ec_005_"+str(i)+".out", 51) for i in range(1,6)]
    output4 = [load_slurm_output(path, game + "_ec_0001_"+str(i)+".out", 51) for i in range(1,6)]
    output5 = [load_slurm_output(path, game + "_ec_0005_"+str(i)+".out", 51) for i in range(1,6)]


    mean_episode_return1 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output1]
    mean_episode_return2 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output2]
    mean_episode_return3 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output3]
    mean_episode_return4 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output4]
    mean_episode_return5 = [np.mean(np.array([x.get("mean_episode_return", 0.0) for x in output.values()])) for output in output5]

    print("mean_return_ec_01: ", np.mean(mean_episode_return1))
    print("mean_return_ec_001: ", np.mean(mean_episode_return2))
    print("mean_return_ec_005: ", np.mean(mean_episode_return3))
    print("mean_return_ec_0001: ", np.mean(mean_episode_return4))
    print("mean_return_ec_0005: ", np.mean(mean_episode_return5))

def tuning_iride(path, game):
    output = [np.array([[get_mean_return_from_dict(load_slurm_output(path + "round" + str(i) + "/", game + "_iride_ilc_" + j + "_ecc_" + z + ".out"))
                for z in ["001", "05", "025", "075", "099"]]
               for j in ["1", "01", "5", "05", "10"]])
              for i in range(1, 4)]

    average_output = (output[0] + output[1] + output[2])/3
    print(average_output)
    # print(output[0])
    # print(output[1])
    # print(output[2])

    label = np.array([[a + " " + b for a in ["001", "05", "025", "075", "099"]] for b in ["1", "01", "5", "05", "10"]])
    print(label)

def get_mean_return_from_dict(dicts):
    return np.mean(np.array([x.get("mean_episode_return", 0.0) for x in dicts.values()]))


def load_slurm_output(path, filename, skip_num=0):
    output_dict = OrderedDict()
    with open(path+filename, "r") as file:
        # skip lines
        num=0
        for _ in file:
            if num == skip_num:
                break
            num += 1
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



print("caveflyer:")
# tuning_learning_rate("data/caveflyer/", "caveflyer")
# tuning_batch_size("data/caveflyer/", "caveflyer")
# tuning_unroll_length("data/caveflyer/", "caveflyer")
# tuning_irc("data/caveflyer/", "caveflyer")
# tuning_ec("data/caveflyer/", "caveflyer")
tuning_iride("data/caveflyer/iride/parameter/", "caveflyer")
print("climber:")
# tuning_learning_rate("data/climber/", "climber")
# tuning_batch_size("data/climber/", "climber")
# tuning_unroll_length("data/climber/", "climber")
# tuning_irc("data/climber/", "climber")
# tuning_ec("data/climber/", "climber")
# tuning_iride("data/climber/iride/parameter/", "cb")
print("coinrun:")
# tuning_learning_rate("data/coinrun/", "coinrun")
# tuning_batch_size("data/coinrun/", "coinrun")
# tuning_unroll_length("data/coinrun/", "coinrun")
# tuning_irc("data/coinrun/", "coinrun")
# tuning_ec("data/coinrun/", "coinrun")
print("jumper:")
# tuning_learning_rate("data/jumper/", "jumper")
# tuning_batch_size("data/jumper/", "jumper")
# tuning_unroll_length("data/jumper/", "jumper")
# tuning_irc("data/jumper/", "jumper")
# tuning_ec("data/jumper/", "jumper")
print("leaper:")
# tuning_learning_rate("data/leaper/", "leaper")
# tuning_batch_size("data/leaper/", "leaper")
# tuning_unroll_length("data/leaper/", "leaper")
# tuning_irc("data/leaper/", "leaper")
# tuning_ec("data/leaper/", "leaper")
print("ninja")
# tuning_learning_rate("data/ninja/", "ninja")
# tuning_batch_size("data/ninja/", "ninja")
# tuning_unroll_length("data/ninja/", "ninja")
# tuning_irc("data/ninja/", "ninja")
# tuning_ec("data/ninja/", "ninja")
