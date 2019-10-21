import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from math import pi
from multiprocessing import Process

fsize = 17

def read_data():
    monte, basin, inter = [], [], []
    paths = ['monte/', 'basin/', 'inter/']
    for i in range(len(paths)):
        if os.path.exists(paths[i]):
            for fi in os.listdir(paths[i]):
                local = []
                with open(paths[i] + fi, 'rb') as f:
                    try:
                        local = pickle.load(f)
                        if i == 0: monte = local #best.append(local)
                        if i == 1: basin = local #basin.append(local)
                        if i == 2: inter = local #inter.append(local)
                    except Exception as e:
                        print(e)


    # data = [all_samples, best_exps, errors, size]
    # best_exps = [best_exp p=1, best_exp p=2, etc...]
    plt.errorbar([x+1 for x in range(len(monte[1]))], monte[1], yerr=monte[2], fmt='--x', color='blue', label='Monte Carlo', zorder=40, capsize=5, linewidth=2)
    plt.errorbar([x+1 for x in range(len(basin[1]))], basin[1], yerr=basin[2], fmt='--^', color='red', label='Basin Hopping', zorder=40, capsize=5, linewidth=2)
    plt.errorbar([x+1 for x in range(len(inter[1]))], inter[1], yerr=inter[2], fmt='--s', color='green', label='Interpolation', zorder=40, capsize=5, linewidth=2)

    # Psuedo-optimal line
    optimal = [0.896574551961, 0.919576599802, 0.938062991772, 0.955755933355, 0.971929679611, 0.981809181437, 0.987384610753, 0.99074475344, 0.99227625948, 0.99339209432]
    plt.plot([x+1 for x in range(len(optimal))], optimal, '--ko', label='Psuedo-optimal')


    plt.legend(fontsize=fsize-2, loc=3)
    plt.gca().set_ylabel('Approximation ratio', fontsize=fsize, labelpad=10)
    plt.gca().set_xlabel('$p$', fontsize=fsize)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=fsize)
    plt.yticks([0.8,0.85,0.9,0.95,1], size=fsize)
    plt.yticks(size=fsize)
    #plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    #plt.gca().set_ylim([0.89, 1])
    #plt.title('$p=$' + str(p))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    read_data()
