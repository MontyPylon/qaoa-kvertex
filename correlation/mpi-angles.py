from mpi4py import MPI
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.optimize import basinhopping
import sys
sys.path.insert(0, '../common/')
import os
import common
import pickle
import random
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def qaoa(gb, G, C, M , k, p):
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def get_angles(G, C, M, k, p):
    # restrict angles to lower left hand corner
    low_g, up_g = 0, 2
    low_b, up_b = 0, 0.3
    bounds = [[low_g, up_g] if j < p else [low_b, up_b] for j in range(2*p)]

    best_angles = []
    best_exp = -1
    num_samples = 20
    for i in range(num_samples):
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=50, disp=False, niter_success=10)
        if -optimal.fun > best_exp:
            best_exp = -optimal.fun
            best_angles = optimal.x
    return best_exp, best_angles

if __name__ == '__main__':
    random.seed(1)
    seed = random.randint(1,1000) + rank
    random.seed(seed)

    num_nodes = 10
    k = int(num_nodes/2)
    G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    C = common.create_C(G, k)
    type = 'complete'
    M = common.create_complete_M(num_nodes, k)
    best_sol = common.brute_force(G, k)

    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.show()

    all_exps, all_angles = [], []
    for p in range(3, 10):
        rank_exp, rank_angles = get_angles(G, C, M, k, p)
        all_exps.append(rank_exp)
        all_angles.append(rank_angles)
        pickle.dump([all_exps, all_angles], open('data-complete-10/' + str(seed) + '.seed', 'wb'))
