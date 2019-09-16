import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from math import pi

def read_data():
    data = []
    #path = 'test-complete-6/4.seed'
    path = 'data-complete-10/138.seed'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(e)

    print(data)
    p = 3
    gamma = data[1][p-3][:p]
    beta = data[1][p-3][p:]
    print(gamma)
    print(beta)

    #plt.plot([x+1 for x in range(p)], gamma, '-o', label='gamma')
    plt.plot([x+1 for x in range(p)], beta, '-o', label='gamma')
    plt.show()

    ''' 
    for i in range(len(lines_g)):
        plt.plot([x+1 for x in range(p)], lines_g[i], '-o', label=str(i))
    plt.xlabel('$\\gamma_i$')
    plt.ylabel('Value of $\\gamma_i$')
    for i in range(len(lines_b)):
        plt.plot([x+1 for x in range(p)], lines_b[i], '-o')
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
