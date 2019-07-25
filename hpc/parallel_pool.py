import pickle
from itertools import combinations
import numpy as np
import networkx as nx
import random
from math import pi
import datetime
from scipy.optimize import basinhopping
from scipy.special import comb
import os
import sys
sys.path.insert(0, '/home/montypylon/lanl/qaoa-kvertex/mixer-phase/')
import common
import time
import multiprocessing
from multiprocessing import Value

G, C, M, k, iterations, bounds, sample_g, sample_b = None, None, None, None, None, None, None, None

def qaoa(gb, *a):
    G, C, M, k, p = a[0], a[1], a[2], a[3], a[4]
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, state)

def brute_force(gi):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
    comb = combinations(G.nodes, k)
    highest = 0
    for group in list(comb):
        score = 0
        for edge in G.edges:
            for v in group:
                if v == edge[0] or v == edge[1]:
                    score += 1
                    break
        if score > highest:
            highest = score
    return highest

def update_best(top_exps, top_angles, opt_exp, opt_angles):
    eps = 0.001
    if not top_angles:
        # first round
        top_exps = opt_exp
        top_angles.append(opt_angles)
    elif abs(opt_exp - top_exps) < eps:
        # same value, check if angles are eps similar
        flag = False
        for j in range(len(top_angles)):
            t = 0
            for l in range(len(opt_angles)):
                t += abs(top_angles[j][l] - opt_angles[l])
            if t < eps:
                flag = True
                break
        if not flag:
            # new set of angles
            top_angles.append(opt_angles)
    elif opt_exp > (top_exps + eps):
        # new maximum
        top_exps, top_angles = opt_exp, []
        top_angles.append(opt_angles)
    return top_exps, top_angles

def multi(i):
    print(i)
    random.seed(1)
    #global G, C, M, k, p, iterations, bounds, sample_g, sample_b
    G, C, M, k = common.get_stuff(91)
    p = 1
    num_samples = 2
    sample_g, sample_b = common.MLHS(p, num_samples, 0, pi/2, 0, pi)
    angles = [sample_g, sample_b]
    #kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': [[0,pi/2],[0,pi]]}
    #optimal = basinhopping(qaoa, [random.uniform(0,pi/2), random.uniform(0,pi)], minimizer_kwargs=kwargs, niter=5, disp=False)
    #return [-optimal.fun, optimal.x]
    exp = 0
    for i in range(20):
        exp = common.expectation(G, qaoa([G, C, M, k, p], angles))
    return exp

def main(gi, p):
    #global G, C, M, k, iterations, bounds, sample_g, sample_b
    #G, C, M, k = common.get_stuff(gi)
    num_samples = 2
    iterations = 5
    sample_g, sample_b = common.MLHS(p, num_samples, 0, pi/2, 0, pi)
    bounds = [[0,pi/2] if i < p else [0,pi] for i in range(2*p)]
    with multiprocessing.Pool() as pool:
        results = pool.map(multi, range(0, num_samples))
    print(results)
    return 0, 0, 0, 0

if __name__ == '__main__':
    print('Number of CPUs: ' + str(os.cpu_count()))
    start = int(round(time.time() * 1000))
    gi = 91
    max_p = 1
    data = [[] for i in range(6)]
    data[0] = gi
    data[1] = brute_force(gi)
    for p in range(1, max_p+1):
        exp, angles, data[4], data[5] = main(gi, p)
        data[2].append(exp)
        data[3].append(angles)
        # data
        print(data)
        pickle.dump(data, open('data/dicke_ps_complete/' + str(gi) + '.data', 'wb'))
        now = int(round(time.time() * 1000))
        print('p: ' + str(p) + '\t t: ' + str((now - start)/1000))
