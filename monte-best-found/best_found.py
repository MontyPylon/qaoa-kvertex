import numpy as np
import networkx as nx
from math import pi
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys
sys.path.insert(0, '../common/')
import os
import common
import pickle
import random
import time
import datetime

def qaoa(gb, G, C, M , k, p):
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def find_best(G, C, M, k, p, num_samples):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    best_exp = -1
    for i in range(num_samples):
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        value = -qaoa(angles, G, C, M, k, p)
        if value > best_exp: best_exp = value
    return best_exp

if __name__ == '__main__':
    # Description: using a constant number of samples, show that the best found solution is decreasing
    # specifically, for an average of x samples (take y of these), the best found solution is decreasing

    num_nodes = 10
    G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    k = int(len(G.nodes)/2)
    C = common.create_C(G, k)
    M = common.create_complete_M(len(G.nodes), k)
    #M = common.create_ring_M(len(G.nodes), k)

    best_sol = common.brute_force(G, k)
    num_samples = 25
    s = 25

    all_samples = []
    best_exps = []
    errors = []
    now = datetime.datetime.now()
    for p in range(1,5):
        print('p = ' + str(p))
        samples = []
        for i in range(s):
            best_exp = find_best(G, C, M, k, p, num_samples)
            samples.append(best_exp/best_sol)
            if i+1 % 5 == 0: print('\ti: ' + str(i) + '\tavg: ' + str(np.average(samples)) \
                                  + '\tstd: ' + str(np.std(samples)) + '\terr: ' + str(2.576*np.std(samples)/np.sqrt(i+1)))
        all_samples.append(samples)
        best_exps.append(np.average(samples))
        errors.append(2.576*np.std(samples)/np.sqrt(s))
        pickle.dump([all_samples, best_exps, errors], open('data/' + str(now), 'wb'))
