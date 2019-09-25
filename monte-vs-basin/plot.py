import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from math import pi
from multiprocessing import Process

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
    plt.errorbar([x+1 for x in range(len(monte[1]))], monte[1], yerr=monte[2], fmt='-o', color='blue', label='Monte Carlo', zorder=40, capsize=5, linewidth=2)
    plt.errorbar([x+1 for x in range(len(basin[1]))], basin[1], yerr=basin[2], fmt='-o', color='red', label='Basin Hopping', zorder=40, capsize=5, linewidth=2)
    plt.errorbar([x+1 for x in range(len(inter[1]))], inter[1], yerr=inter[2], fmt='-o', color='green', label='Interpolation', zorder=40, capsize=5, linewidth=2)


    plt.legend(fontsize=17)
    plt.gca().set_ylabel('Approximation ratio', fontsize=17, labelpad=10)
    plt.gca().set_xlabel('$p$', fontsize=17)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=17)
    #plt.yticks([0.8,0.85,0.9,0.95,1], size=17)
    plt.yticks(size=17)
    #plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    #plt.gca().set_ylim([0.89, 1])
    #plt.title('$p=$' + str(p))
    plt.tight_layout()
    plt.show()


    #plt.xticks([x+1 for x in range(p)])
    #plt.legend()
    #plt.gca().set_ylim([0, 0.16])
    #plt.xticks(np.arange(1, , step=1))
    #plt.tight_layout()
    #plt.title('$p=$' + str(p))
    #plt.show()

if __name__ == '__main__':
    read_data()
