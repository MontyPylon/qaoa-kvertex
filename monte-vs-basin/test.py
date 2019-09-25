import numpy as np
import networkx as nx
from math import pi
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys
sys.path.insert(0, '../common/')
sys.path.insert(0, './')
import methods
import os
import common
import pickle
import random
import time
import datetime

if __name__ == '__main__':
    random.seed(6)

    num_nodes = 6
    G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    k = int(len(G.nodes)/2)
    C = common.create_C(G, k)
    M = common.create_complete_M(len(G.nodes), k)
    #M = common.create_ring_M(len(G.nodes), k)

    best_sol = common.brute_force(G, k)
    num_samples = 100
    s = 25
    z_star = 1.96

    all_samples = []
    best_exps = []
    errors = []
    now = datetime.datetime.now()
    for p in range(1,11):
        print('p = ' + str(p))
        samples = []
        for i in range(s):
            best_exp = methods.inter_best(G, C, M, k, p, num_samples)
            samples.append(best_exp/best_sol)
            if (i+1) % 5 == 0: print('\ti: ' + str(i+1) + '\tavg: ' + str(np.average(samples)) \
                                  + '\tstd: ' + str(np.std(samples)) + '\terr: ' + str(z_star*np.std(samples)/np.sqrt(i+1)))
        all_samples.append(samples)
        best_exps.append(np.average(samples))
        errors.append(z_star*np.std(samples)/np.sqrt(s))
        pickle.dump([all_samples, best_exps, errors], open('best_found/' + str(now), 'wb'))
