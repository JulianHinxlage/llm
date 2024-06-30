
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def sub_plot(history, name):
    if len(history) > 0:
        values = np.array(history).transpose()
        values[0] = np.cumsum(values[0])

        move = False
        if move:
            max = values[0, -1]
            values[0] -= max

        average = True
        if average:

            x =  np.array(values[0])
            if len(values[1]) >= 20:
                y = moving_average(values[1], 20)
            else:
                y = []
            off = len(x) - len(y)
            x = x[off:]
        else:
            x = values[0]
            y = values[1]


        plt.plot(x, y, label=name)

def plot(name, log_scale=False):
    plt.xlabel('Total Tokens')
    if log_scale:
        plt.yscale('log')
    plt.ylabel(name)
    plt.title(name)
    plt.legend()
    plt.show()

def add_model(file, name, metric, metric2=""):
    if os.path.exists(file):
        state = torch.load(file)
        if 'parameters' in state:
            if 'description' in state['parameters']:
                desc = state['parameters']['description']
                if desc != "":
                    name = f"[{desc}] " + name
        if 'name' in state:
            name = f"[{state['name']}] " + name
        if metric in state['statistics']:
            sub_plot(state['statistics'][metric], name + f" ({metric})")
        if metric2 in state['statistics']:
            sub_plot(state['statistics'][metric2], name + f" ({metric2})")

def main():
    metric = 'loss'
    metric = 'perplexity'
    metric = 'accuracy'
    add_model(f"models/91M/v1/model.dat", "v4", metric, 'val_' + metric)
    plot(metric, metric == 'loss')

if __name__ == "__main__":
    main()
