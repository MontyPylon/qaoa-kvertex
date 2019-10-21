from mpi4py import MPI
import numpy as np
import networkx as nx
from math import pi
from scipy.optimize import basinhopping
from scipy.special import comb
import sys
sys.path.insert(0, '../common/')
#sys.path.insert(0, './')
import os
import common
import pickle
import random
import time
import datetime
#import methods

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()


def qaoa(gb, G, C, M , k, p):
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def qaoa_initial(gb, G, C, M , k, p, m):
    size = int(comb(len(G.nodes), k))
    state = np.zeros(size)
    state[m] = 1
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def initial(G, C, M, k, p, num_samples):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
    size = int(comb(len(G.nodes), k))
    m = random.randint(0, size - 1)
    evals = 0
    n_iter = 10
    max_fun = num_samples/10
    best_exp = -1
    while evals < num_samples:
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        kwargs = {'method': 'L-BFGS-B', 'options': {'maxfun': max_fun}, 'args': (G, C, M, k, p, m), 'bounds': bounds}
        optimal = basinhopping(qaoa_initial, angles, minimizer_kwargs=kwargs, niter=n_iter, disp=False)
        evals += optimal.nfev
        if -optimal.fun > best_exp: best_exp = -optimal.fun
    return best_exp

def dicke(G, C, M, k, p, num_samples):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
    evals = 0
    n_iter = 10
    max_fun = num_samples/10
    best_exp = -1
    while evals < num_samples:
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        kwargs = {'method': 'L-BFGS-B', 'options': {'maxfun': max_fun}, 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=n_iter, disp=False)
        evals += optimal.nfev
        if -optimal.fun > best_exp: best_exp = -optimal.fun
    return best_exp

if __name__ == '__main__':
    if len(sys.argv) != 3:
        if rank == 0: print('Need proper number of arguments')
        sys.exit(1)

    # arguments: [graph_seed] [collection_type]
    # collection types:
    # 1 = dicke_complete
    # 2 = dicke_ring
    # 3 = k_complete
    # 4 = k_ring
    graph_seed = int(sys.argv[1])
    arg = sys.argv[2]
    if arg != '1' and arg != '2' and arg != '3' and arg != '4':
        if rank == 0: print('arg options are 1,2,3,4')
        sys.exit(1)
    arg = int(arg)

    random.seed(graph_seed) # create same graph on all cpus
    num_nodes = 7
    G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    k = int(len(G.nodes)/2)
    C = common.create_C(G, k)
    M = None
    if arg == 1 or arg == 3:
        M = common.create_complete_M(len(G.nodes), k)
    else:
        M = common.create_ring_M(len(G.nodes), k)

    # diverge random seed on cpus so samples are not the same
    seed = random.randint(1,1000) + rank
    random.seed(seed)
    best_sol = common.brute_force(G, k)
    base_num_samples = 5000 # collect x samples on each cpu, remember size = # cpus
    z_star = 1.96 # confidence interval of 1.96 = 95%, 2.56 = 99%

    all_samples, max_exps, errors = [], [], []
    now = datetime.datetime.now()

    for p in range(1,11):

        best_exp = None
        num_samples = base_num_samples*p
        # dicke complete and dicke ring
        if arg == 1 or arg == 2: best_exp = dicke(G, C, M, k, p, num_samples)
        # k complete and k ring
        if arg == 3 or arg == 4: best_exp = initial(G, C, M, k, p, num_samples)

        # scale to approximation ratio
        best_exp = best_exp/best_sol

        samples = None
        if rank == 0: samples = np.empty(size, dtype='d')
        comm.Gather(best_exp, samples, root=0)

        if rank == 0:
            #print('round ' + str(p) + ': ' + str(samples))
            all_samples.append(samples)
            if arg == 1 or arg == 2:
                max_exps.append(np.max(samples))
            else:
                max_exps.append(np.average(samples))
            errors.append(z_star*np.std(samples)/np.sqrt(size))
            print('p = ' + str(p) + '\tmax_exps: ' + str(max_exps))
            print('\terrors:   ' + str(errors))
            folder = None
            if arg == 1: folder = 'dicke_complete/'
            if arg == 2: folder = 'dicke_ring/'
            if arg == 3: folder = 'k_complete/'
            if arg == 4: folder = 'k_ring/'
            if not os.path.exists(folder): os.mkdir(folder)
            pickle.dump([all_samples, max_exps, errors, size], open(folder + str(graph_seed), 'wb'))
