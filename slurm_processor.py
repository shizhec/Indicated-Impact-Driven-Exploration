from collections import OrderedDict
import csv


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


def dict_to_csv(path, filename, skip_num=0):
    out_dict = load_slurm_output(path, filename+".out", skip_num)

    dict_list = []
    for frame, state_dict in out_dict.items():
        state_dict['frame']=frame
        dict_list.append(state_dict)

    header = dict_list[0].keys()
    with open(path+filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)


for i in range(1, 11):
    dict_to_csv("data/ninja/iride/sample-efficiency/", str(i), 49)