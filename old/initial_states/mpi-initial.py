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
    G, C, M, k, p = a[0], a[1], a[2], a[3], a[4]
    state = common.random_k_state(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, state)

def work(gi, p, s):
    G, C, M, k = common.get_stuff(gi)
    sample_g, sample_b = common.MLHS(p, s, 0, pi/2, 0, pi)
    bounds = [[0,pi/2] if j < p else [0,pi/2] for j in range(2*p)]
    data = []
    for i in range(s):
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=0, disp=False)
        data.append(-optimal.fun)
    return data

if __name__ == '__main__':
    random.seed(10 + rank)
    gi = random.randint(163,955)
    print('aquired gi: ' + str(gi) + ' from: ' + str(rank))

    s_per_rank = 2
    max_p = 2
    z = 2.576 # z* for 99% confidence interval

    max_exp, max_std, error = [], [], []
    for p in range(1, max_p+1):
        # do work over indices
        best_exps = work(gi, p, s_per_rank)
        print(best_exps)
        brute = common.brute_force(gi)
        best = np.average(best_exps)
        std = np.std(best_exps)
        approx = best/brute
        max_exp.append(approx)
        error.append(z*np.std(best_exps)/np.sqrt(s_per_rank))
        print('gi: ' + str(gi) + ', max_exp: ' + str(max_exp) + ', error: ' + str(error))
        pickle.dump([max_exp, error], open('init-complete/' + str(gi) + '.mpi', 'wb'))
        #pickle.dump([gi, [i+1 for i in range(p)], max_exp, max_std, error, s_per_rank, size], open('data/' + str(gi) + '.mpi-k', 'wb'))
