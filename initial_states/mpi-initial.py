from mpi4py import MPI
import numpy as np
import networkx as nx
from math import pi
import random
import time
from scipy.special import comb
from scipy.optimize import basinhopping
import sys
sys.path.insert(0, '../mixer-phase/')
import common
import dicke_ps_complete
import pickle

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()


def qaoa(gb, *a):
    G, C, M, k, p, m = a[0], a[1], a[2], a[3], a[4], a[5]
    state = common.select_k_state(len(G.nodes), k, m)
    state = common.mixer(state, M, pi/2)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, state)

def work(gi, p, s):
    G, C, M, k = common.get_stuff(gi)
    sample_g, sample_b = common.MLHS(p, s, 0, pi/2, 0, pi)
    bounds = [[0,pi/2] if j < p else [0,pi] for j in range(2*p)]
    best_avg = -1
    for i in range(s):
        exp, data = -1, []
        for m in range(int(comb(len(G.nodes), k))):
            kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p, m), 'bounds': bounds}
            optimal = basinhopping(qaoa, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=0, disp=False)
            if -optimal.fun > exp: exp = -optimal.fun
            data.append(exp)
        if np.average(data) > best_avg: best_avg = np.average(data)
    return best_avg

if __name__ == '__main__':
    #indices = np.linspace(rank*s_per_rank, (rank+1)*s_per_rank-1, s_per_rank)
    gi = 91
    s_per_rank = 2
    z = 2.576 # z* for 99% confidence interval
    max_p = 1

    max_exp, max_std, error = [], [], []
    for p in range(1, max_p+1):
        # do work over indices
        rank_exp = work(gi, p, s_per_rank)

        # gather best exp
        data = None
        if rank == 0: data = np.empty(size, dtype='d')
        comm.Gather(rank_exp, data, root=0)

        if rank == 0:
            best = common.brute_force(gi)
            data = [x/best for x in data]
            max_exp.append(np.average(data))
            max_std.append(np.std(data))
            error.append(z*np.std(data)/np.sqrt(size))
            pickle.dump([gi, [i+1 for i in range(p)], max_exp, max_std, error, s_per_rank, size], open('data/' + str(gi) + '.mpi-k', 'wb'))
