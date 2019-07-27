from mpi4py import MPI
import numpy as np
import networkx as nx
from math import pi
import random
import time
import sys
sys.path.insert(0, '../mixer-phase/')
import common
import dicke_ps_complete
import pickle

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

def work(gi, p, s):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    best_exp = -1
    for j in range(s):
        angles = [random.uniform(0,pi/2) if i < p else random.uniform(0,pi) for i in range(2*p)]
        exp = common.expectation(G, dicke_ps_complete.qaoa([G, C, M, k, p], angles))
        if exp > best_exp: best_exp = exp
    print('rank: ' + str(rank) + ', best_exp: ' + str(best_exp))
    return best_exp

if __name__ == '__main__':
    #indices = np.linspace(rank*s_per_rank, (rank+1)*s_per_rank-1, s_per_rank)
    gi = 91
    s_per_rank = 10
    z = 2.576 # z* for 99% confidence interval
    max_p = 7

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
            pickle.dump([gi, [i+1 for i in range(p)], max_exp, max_std, error, s_per_rank, size], open('data/' + str(gi) + '.mpi-carlo', 'wb'))
