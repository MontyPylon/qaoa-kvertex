from mpi4py import MPI
import numpy as np
from math import pi
from scipy.optimize import basinhopping
import sys
sys.path.insert(0, '../mixer-phase/')
import os
import common
import pickle
import random
import time

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
    sample_g, sample_b = common.MLHS(p, s, 0, pi/2, 0, pi)
    bounds = [[0,pi/2] if j < p else [0,pi] for j in range(2*p)]
    eps = 0.001
    best, angles = -1, []
    for i in range(s):
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=2, disp=False)
        if -optimal.fun > best + eps:
            best = -optimal.fun
            angles = [list(optimal.x)]
        elif -optimal.fun > best - eps:
            angles.append(list(optimal.x))
    return best, angles

if __name__ == '__main__':
    for i in range(1000):
        gi = None
        if rank == 0:
            gi = random.randint(1, 955)
            while os.path.exists('data/' + str(gi) + '.angles'):
                print(gi)
                gi = random.randint(1,955)

        gi = comm.bcast(gi, root=0)

        s_per_rank = 10
        max_p = 3
        eps = 0.001

        max_exp, max_angles = [], []
        for p in range(1, max_p+1):
            # do work over indices
            rank_exp, rank_angles = work(gi, p, s_per_rank)
            # gather best exp
            data = comm.gather(rank_exp, root=0)
            angles = comm.gather(rank_angles, root=0)

            if rank == 0:
                # sort and trim the fat
                ranks = [i for i in range(size)]
                y = [x for _,x in sorted(zip(data, ranks))]
                data = sorted(data)
                y.reverse()
                data.reverse()
                best_exp = data[0]
                best_angles = angles[y[0]]
                for i in range(len(data)-1):
                    if abs(best_exp - data[i+1]) < eps: best_angles.extend(angles[y[i+1]])
                    else: break

                max_exp.append(best_exp)
                max_angles.append(best_angles)
                pickle.dump([max_exp, max_angles], open('data/' + str(gi) + '.angles', 'wb'))
