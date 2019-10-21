import numpy as np
import networkx as nx
from math import pi
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys
sys.path.insert(0, '../common/')
import os
import matplotlib.pyplot as plt
import common
import pickle
import random
import time
import datetime

def find_best(monte, p, num_samples, i):
    lower = int(num_samples*i)
    upper = int(num_samples*(i+1))
    sample = monte[0][p-1][lower:upper]
    return np.max(sample)

if __name__ == '__main__':
    monte = pickle.load(open('dist/2019-09-08 10:33:30.502320', 'rb'))
    #monte = pickle.load(open('dist/2019-09-08 13:28:50.508902', 'rb'))
    num_samples = len(monte[0][0]) / 36
    s = 36

    all_samples = []
    best_exps = []
    errors = []
    now = datetime.datetime.now()
    for p in range(1,11):
        print('p = ' + str(p))
        samples = []
        for i in range(s):
            best_exp = find_best(monte, p, num_samples, i)
            samples.append(best_exp)
            if (i+1) % 5 == 0: print('\ti: ' + str(i+1) + '\tavg: ' + str(np.average(samples)) \
                                  + '\tstd: ' + str(np.std(samples)) + '\terr: ' + str(1.96*np.std(samples)/np.sqrt(i+1)))
        all_samples.append(samples)
        best_exps.append(np.average(samples))
        errors.append(1.96*np.std(samples)/np.sqrt(s))

    font_size = 20
    plt.errorbar([x+1 for x in range(len(best_exps))], best_exps, yerr=errors, fmt='--bo', capsize=5)
    #plt.legend(loc=4, fontsize=17)
    plt.gca().set_ylabel('Best approximation ratio', fontsize=font_size, labelpad=15)
    plt.gca().set_xlabel('$p$', fontsize=font_size)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=font_size)
    #plt.yticks([0.005, 0.01, 0.015, 0.02, 0.025], size=17)
    plt.yticks(size=font_size)
    plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    #plt.gca().set_ylim([0,0.03])
    plt.tight_layout()
    plt.show()
