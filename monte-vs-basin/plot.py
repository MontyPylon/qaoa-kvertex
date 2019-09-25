import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from math import pi
from multiprocessing import Process

def read_data():
    best, basin, inter = [], [], []
    paths = ['best_found/', 'basin/', 'interpolation/']
    for i in range(len(paths)):
        if os.path.exists(paths[i]):
            for fi in os.listdir(paths[i]):
                local = []
                with open(paths[i] + fi, 'rb') as f:
                    try:
                        local = pickle.load(f)
                        if i == 0: best.append(local)
                        if i == 1: basin.append(local)
                        if i == 2: inter.append(local)
                    except Exception as e:
                        print(e)


    for i in range(len(best)):
        # best = [all_samples, best_exps, errors]
        # best_exps = [best_exp p=1, best_exp p=2, etc...]
        plt.plot([x+1 for x in range(len(best[i][1]))], best[i][1], '-o', color='blue', label='best_found')


    #for i in range(len(beta)):
    #    plt.plot([x+1 for x in range(p)], beta[i], '-o', color='lightgrey', label='beta')

    #plt.errorbar([x+1 for x in range(p)], avg_gamma, yerr=std_gamma, fmt='-o', color='red', label='avg_gamma', zorder=40, capsize=5, linewidth=2)
    #plt.errorbar([x+1 for x in range(p)], avg_beta, yerr=std_beta, fmt='-o', color='red', label='avg_beta', zorder=40, capsize=5, linewidth=2)
    plt.legend()
    plt.show()


    

    '''
    plt.xlabel('$\\beta_i$')
    plt.ylabel('Value of $\\beta_i$')

    plt.errorbar([x+1 for x in range(len(monte[0]))], monte[0], yerr=monte[1], fmt='--ro', capsize=5)
    #plt.legend(loc=4, fontsize=17)
    plt.gca().set_ylabel('Number of samples', fontsize=17, labelpad=10)
    plt.gca().set_xlabel('$p$', fontsize=17)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=17)
    #plt.yticks([0.8,0.85,0.9,0.95,1], size=17)
    plt.yticks(size=17)
    plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    plt.yscale('log')
    #plt.gca().set_ylim([0.89, 1])
    #plt.title('$p=$' + str(p))
    plt.tight_layout()
    plt.show()
    '''

    #plt.xticks([x+1 for x in range(p)])
    #plt.legend()
    #plt.gca().set_ylim([0, 0.16])
    #plt.xticks(np.arange(1, , step=1))
    #plt.tight_layout()
    #plt.title('$p=$' + str(p))
    #plt.show()

if __name__ == '__main__':
    read_data()
