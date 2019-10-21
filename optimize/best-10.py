from mpi4py import MPI
import networkx as nx
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

def get_angles(G, C, M, k, p, best_sol):

    # restrict angles to lower left hand corner
    low_g, up_g = 0, 2.1
    low_b, up_b = 0, 0.3
    bounds = [[low_g, up_g] if j < p else [low_b, up_b] for j in range(2*p)]

    best_angles = []
    best_exp = -1
    max_fun = 100
    num_samples = 1

    for i in range(num_samples):
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        kwargs = {'method': 'L-BFGS-B', 'options': {'maxfun': max_fun}, 'args': (G, C, M, k, p), 'bounds': bounds}
        #kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=0, disp=False)
        if -optimal.fun > best_exp:
            best_exp = -optimal.fun
            best_angles = optimal.x
            print('p = 1 results')
            print('approx: ' + str(best_exp/best_sol) + '\tangles: ' + str(best_angles))
    return best_exp, best_angles

def get_angles(G, C, M, k, p, best_sol, lin):
    for i in range(num_samples):
        angles = []
        low_g = random.uniform(0.35, 0.55)
        high_g = random.uniform(1, 1.3)
        low_b = random.uniform(0.03, 0.055)
        high_b = random.uniform(0.065, 0.08)
        gamma = np.linspace(low_g, high_g, p)
        gamma = list(gamma)
        beta = np.linspace(high_b, low_b, p)
        beta = list(beta)
        angles = gamma.copy()
        angles.extend(beta)
        #print('angles: ' + str(angles))
        #angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        kwargs = {'method': 'L-BFGS-B', 'options': {'maxfun': max_fun}, 'args': (G, C, M, k, p), 'bounds': bounds}
        #kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=0, disp=False)
        if -optimal.fun > best_exp:
            best_exp = -optimal.fun
            best_angles = optimal.x
            print('approx: ' + str(best_exp/best_sol) + '\tangles: ' + str(best_angles))
    return best_exp, best_angles

if __name__ == '__main__':
    #random.seed(82)
    #seed = random.randint(1,1000) + rank
    seed = 1
    random.seed(seed)
    print('rank: ' + str(rank) + ', seed: ' + str(seed))

    num_nodes = 6
    k = int(num_nodes/2)
    G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    C = common.create_C(G, k)
    M = common.create_complete_M(num_nodes, k)
    best_sol = common.brute_force(G, k)
    s = random.randint(1,1000) + rank
    random.seed(s)

    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.show()

    all_exps, all_angles = [], []
    lin = [] #[low_g guess, high_g guess, high_b guess, low_b guess]
    prev_angles = []
    for p in range(1, 2):
        if p == 1 or p == 2: rank_exp, rank_angles = get_angles(G, C, M, k, p, best_sol)
        else: rank_exp, rank_angles = get_angles(G, C, M, k, p, best_sol, lin)

        # find largest exp, and broadcast it's set of angles to every rank
        exps = None
        if rank == 0: exps = np.empty(size, dtype='d') # d for double
        exps = comm.gather(rank_exp, root=0)
        max_index = None
        if rank == 0:
            print(exps)
            m = -1
            for i in range(len(exps)):
                if exps[i] > m: max_index = i
        max_index = comm.bcast(max_index, root=0)
        angles = None
        if rank == max_index: angles = rank_angles
        angles = comm.bcast(angles, root=max_index)

        if p == 2:


        #all_exps.append(rank_exp/best_sol)
        #all_angles.append(rank_angles)
        #pickle.dump([G, all_exps, all_angles], open('inter/complete-' + str(num_nodes) + '-' + str(seed) + '.seed', 'wb'))
