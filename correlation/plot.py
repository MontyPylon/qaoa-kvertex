import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from math import pi
from multiprocessing import Process

def read_data():
    data = []
    #path = 'data/complete-6/'
    path = 'interpolation/complete-6/'
    if os.path.exists(path):
        for i in range(2000,5000):
            local = []
            fi = path + str(i) + '.seed'
            if os.path.exists(fi):
                with open(fi, 'rb') as f:
                    try:
                        local = pickle.load(f)
                        data.append(local)
                    except Exception as e:
                        print(e)
    final = []
    #for p in range(3, 9):
    p = 8
    gamma, beta = [], []
    for i in range(len(data)):
        gamma.append(data[i][1][p-3][:p])
        beta.append(data[i][1][p-3][p:])

    '''
    # remove gamma outliers
    for i in range(len(data)-1, -1, -1):
        if gamma[i][0] > 1.5:
            del gamma[i]
            continue

    # remove beta outliers
    for i in range(len(data)-1, -1, -1):
        if beta[i][0] < 0.1:
            del beta[i]
            continue
        for j in range(p):
            if beta[i][j] > 0.25:
                del beta[i]
                continue
    '''

    # average
    avg_gamma, std_gamma = [], []
    avg_beta, std_beta = [], []
    for i in range(p):
        tmp_g, tmp_b = [], []
        for j in range(len(gamma)):
            tmp_g.append(gamma[j][i])
        for j in range(len(beta)):
            tmp_b.append(beta[j][i])
        avg_gamma.append(np.average(tmp_g))
        std_gamma.append(np.std(tmp_g))
        avg_beta.append(np.average(tmp_b))
        std_beta.append(np.std(tmp_b))

    print('avg_gamma: ' + str(avg_gamma))
    print('std_gamma:' + str(np.average(std_gamma)))
    print('avg_beta: ' + str(avg_beta))
    print('std_beta: ' + str(np.average(std_beta)))

    #final.append([avg_gamma, std_gamma, avg_beta, std_beta])
    # save to file
    #print(final)
    #pickle.dump(final, open('interpolation/complete-6.angles', 'wb'))


    for i in range(len(gamma)):
        plt.plot([x+1 for x in range(p)], gamma[i], '-o', color='lightgrey', label='gamma')
    plt.errorbar([x+1 for x in range(p)], avg_gamma, yerr=std_gamma, fmt='-o', color='red', label='avg_gamma', zorder=40, capsize=5, linewidth=2)

    #for i in range(len(beta)):
    #    plt.plot([x+1 for x in range(p)], beta[i], '-o', color='lightgrey', label='beta')
    #plt.errorbar([x+1 for x in range(p)], avg_beta, yerr=std_beta, fmt='-o', color='red', label='avg_beta', zorder=40, capsize=5, linewidth=2)

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
