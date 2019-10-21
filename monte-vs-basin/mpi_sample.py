from mpi4py import MPI
import numpy as np
import networkx as nx
from math import pi
import sys
sys.path.insert(0, '../common/')
sys.path.insert(0, './')
import os
import common
import pickle
import random
import time
import datetime
import methods

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        if rank == 0: print('Need a sys argument')
        sys.exit(1)

    # arguments:
    # 1 = monte
    # 2 = basin
    # 3 = inter
    # 4 = inter2
    arg = sys.argv[1]
    if arg != '1' and arg != '2' and arg != '3' and arg != '4':
        if rank == 0: print('arg options are 1,2,3,4')
        sys.exit(1)
    arg = int(arg)

    random.seed(1) # create same graph on all cpus
    num_nodes = 6
    G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
    k = int(len(G.nodes)/2)
    C = common.create_C(G, k)
    M = common.create_complete_M(len(G.nodes), k)
    #M = common.create_ring_M(len(G.nodes), k)

    # diverge random seed on cpus so samples are not the same
    seed = random.randint(1,1000) + rank
    random.seed(seed)
    best_sol = common.brute_force(G, k)
    num_samples = 100 # collect x samples on each cpu, remember size = # cpus
    z_star = 1.96 # confidence interval of 1.96 = 95%, 2.56 = 99%

    all_samples = []
    best_exps = []
    errors = []
    now = datetime.datetime.now()

    for p in range(1,3):

        best_exp = None

        # monte carlo best found
        if arg == 1: best_exp = methods.monte_best(G, C, M, k, p, num_samples)
        # basin hopping
        if arg == 2: best_exp = methods.basin_best(G, C, M, k, p, num_samples)
        # linear interpolation
        if arg == 3: best_exp = methods.inter_best(G, C, M, k, p, num_samples)
        if arg == 4: best_exp = methods.inter_best2(G, C, M, k, p, num_samples)

        # scale to approximation ratio
        best_exp = best_exp/best_sol

        samples = None
        if rank == 0: samples = np.empty(size, dtype='d')
        comm.Gather(best_exp, samples, root=0)

        if rank == 0:
            print('round ' + str(p) + ': ' + str(samples))
            all_samples.append(samples)
            best_exps.append(np.average(samples))
            errors.append(z_star*np.std(samples)/np.sqrt(size))
            folder = None
            if arg == 1: folder = 'monte/'
            if arg == 2: folder = 'basin/'
            if arg == 3: folder = 'inter/'
            if arg == 4: folder = 'inter2/'
            pickle.dump([all_samples, best_exps, errors, size], open(folder + str(now), 'wb'))
