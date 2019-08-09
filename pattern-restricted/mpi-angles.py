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
size = comm.Get_size()
rank = comm.Get_rank()

def qaoa(gb, *a):
    G, C, M, k, p = a[0], a[1], a[2], a[3], a[4]
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, state)

def work(gi, p, s):
    random.seed(random.randint(1,10000) + rank)
    G, C, M, k = common.get_stuff(gi)
    sample_g, sample_b = common.MLHS(p, s, 0, 0.6, 2.9, pi)
    bounds = [[0,0.6] if j < p else [2.9,pi] for j in range(2*p)]
    eps = 0.001
    best, angles = -1, []
    for i in range(s):
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=0, disp=False)
        if -optimal.fun > best + eps:
            best = -optimal.fun
            angles = [list(optimal.x)]
        elif -optimal.fun > best - eps:
            angles.append(list(optimal.x))
    #print(str(rank) + ', ' + str(best) + ', ' + str(angles))
    return best, angles

def remove_dup(angles):
    final = []
    eps = 0.001
    for i in range(len(angles)):
        if not final: 
            final.append(angles[i])
        else:
            flag = False
            for j in range(len(final)):
                total = 0
                for k in range(len(final[0])):
                    total += abs(angles[i][k] - final[j][k])
                if total < eps:
                    flag = True
                    break
            if not flag:
                final.append(angles[i])
    return final

if __name__ == '__main__':
    for i in range(1000):
        gi = None
        if rank == 0:
            gi = random.randint(1, 955)
            while os.path.exists('data/' + str(gi) + '.angles'):
                gi = random.randint(1,955)

        gi = comm.bcast(gi, root=0)
        #if rank == 0: print('gi = ' + str(gi))

        s_per_rank = 2
        max_p = 6
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
                best_angles = remove_dup(best_angles)
                max_angles.append(best_angles)
                #print(best_exp)
                #for j in range(len(best_angles)): print(best_angles[j])
                pickle.dump([max_exp, max_angles], open('data/' + str(gi) + '.angles', 'wb'))
