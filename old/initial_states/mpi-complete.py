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
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, state)

def work(gi, p, s):
    G, C, M, k = common.get_stuff(gi)
    sample_g = np.linspace(1.45, 1.25, num=p+2)
    sample_g = sample_g[1:p+1]
    sample_b = np.linspace(0.14, 0, num=p+2)
    sample_b = sample_b[1:p+1]
    samples = np.append(sample_g, sample_b)

    bounds = [[0,pi/2] if j < p else [0,pi/2] for j in range(2*p)]
    kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
    optimal = basinhopping(qaoa, samples, minimizer_kwargs=kwargs, niter=s, disp=False)
    return -optimal.fun

if __name__ == '__main__':
    random.seed(10 + rank)
    gi = random.randint(163,955)
    print('aquired gi: ' + str(gi) + ' from: ' + str(rank))

    s_per_rank = 3
    max_p = 6

    max_exp, max_std, error = [], [], []
    for p in range(1, max_p+1):
        # do work over indices
        best_exp = work(gi, p, s_per_rank)
        brute = common.brute_force(gi)
        approx = best_exp/brute
        max_exp.append(approx)
        print('gi: ' + str(gi) + ', max_exp: ' + str(max_exp))
        pickle.dump(max_exp, open('complete/' + str(gi) + '.mpi', 'wb'))
        #error.append(z*np.std(data)/np.sqrt(size))
        #pickle.dump([gi, [i+1 for i in range(p)], max_exp, max_std, error, s_per_rank, size], open('data/' + str(gi) + '.mpi-k', 'wb'))
