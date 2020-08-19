import pickle
from collections import OrderedDict

def plot_normalized_mean_return(Rmax, Rmin, env):
    output = [tuple()]

    last_tuple = tuple()
    while True:
        try:
            obj = pickle.load(open('data/'+env+'.pkl', 'rb'))
            frames = obj.get("frames")
            exp = lambda x: x if x is not "nan" else 0
            episode_mean_return = exp(obj.get("mean_episode_return"))
            current_tuple = (frames, episode_mean_return)
            if not current_tuple == last_tuple:
                output.append(current_tuple)
            last_tuple = current_tuple
        except EOFError:
            break

    print(output)

plot_normalized_mean_return(0,0,"procgen:procgen-coinrun-v0")

