import numpy as np
from math import pi
import networkx as nx
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys
sys.path.insert(0, '../common/')
import os
import common
import pickle
import random
import time

def qaoa(gb, G, C, M , k, p):
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def get_angles(G, C, M, k, p):
    # restrict angles to lower left hand corner
    low_g, up_g = 0, pi
    low_b, up_b = 0, pi/4
    bounds = [[low_g, up_g] if j < p else [low_b, up_b] for j in range(2*p)]

    best_angles = []
    best_exp = -1
    num_samples = 20
    for i in range(num_samples):
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=50, disp=True, niter_success=10)
        if -optimal.fun > best_exp:
            best_exp = -optimal.fun
            best_angles = optimal.x
            print(str(best_exp) + '\t' + str(best_angles))
    return best_exp, best_angles

if __name__ == '__main__':
    seed = 4
    random.seed(seed)

    # Initialize variables
    num_nodes = 6
    k = int(num_nodes/2)
    G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    C = common.create_C(G, k)
    type = 'complete'
    M = common.create_complete_M(num_nodes, k)
    best_sol = common.brute_force(G, k)

    all_exps, all_angles = [], []
    for p in range(3, 7):
        best_exp, best_angles = get_angles(G, C, M, k, p)
        all_exps.append(best_exp/best_sol)
        all_angles.append(best_angles)
        pickle.dump([all_exps, all_angles], open('test-complete-6/' + str(seed) + '.seed', 'wb'))
