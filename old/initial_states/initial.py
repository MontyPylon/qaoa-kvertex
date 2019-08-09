import matplotlib.pyplot as plt
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
import dicke_ps_complete

def random_k_state(n, k):
    state = np.zeros(2**n)
    num_k = int(comb(n, k))
    index = random.randint(0, num_k-1)
    counter = 0
    for i in range(0, 2**n):
        if common.num_ones(i) == k:
            # start with different initial states
            if index == counter:
                state[i] = 1
                break
            counter += 1
    return state

def qaoa(gb, *a):
    G, C, M, k, p = a[0], a[1], a[2], a[3], a[4]
    state = random_k_state(len(G.nodes), k)
    state = common.mixer(state, M, random.uniform(0,pi))
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, state)

#def opt(args, *a):
#    state = qaoa(a, args)
#    return -common.expectation(a[0], state)

def classical(gi, p, s):
    G, C, M, k = common.get_stuff(gi)
    data = []
    num_trials = 20
    n_iter = 2
    for i in range(s):
        print('\ts = ' + str(i) + '\t' + str(datetime.datetime.now().time()))
        sample_g, sample_b = common.MLHS(p, num_trials, 0, pi/2, 0, pi)
        init = [[0,pi/2] if i < p else [0,pi] for i in range(2*p)]
        exp = -1
        for i in range(num_trials):
            kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': init}
            optimal = basinhopping(qaoa, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=n_iter, disp=False)
            if -optimal.fun > exp:
                exp = -optimal.fun
        data.append(exp)
    return data

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

def gather_classical():
    max_p = 10

    s = 25
    z = 2.576 # z* for 99% confidence interval

    #for i in range(1, 1000):
    #    gi = random.randint(31, 995)
    #    print('i, gi = ' + str(i) + ', ' + str(gi) + '\t' + str(datetime.datetime.now().time()))
    gi = 91

    avg, std, error = [], [], []
    for p in range(1, max_p+1):
        print('p = ' + str(p) + '\t' + str(datetime.datetime.now().time()))
        data = classical(gi, p, s)
        best = brute_force(gi)
        data = [y/best for y in data]
        avg.append(np.average(data))
        std.append(np.std(data))
        error.append(z*np.std(data)/np.sqrt(s))
        x = [i for i in range(1, p+1)]
        print([gi, x, avg, std, error, s])
        pickle.dump([gi, x, avg, std, error, s], open('data/' + str(gi) + '.carlo', 'wb'))

def gather_dicke():
    max_p = 7
    x = [i for i in range(1, max_p+1)]
    gi = 91

    approx = [8.503162903355406, 8.800506134417677, 8.952913838859514, 8.984216297214182, 8.993237586035384, 8.998175490445554, 8.99978203457083]
    '''
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)
    for p in range(1, max_p+1):
        print('\tp = ' + str(p) + '\t' + str(datetime.datetime.now().time()))
        data = dicke_ps_complete.
        best = brute_force(gi)
        data = [x/best for x in data]
        max_exp.append(np.average(data))
        max_std.append(np.std(data))
        error.append(z*np.std(data)/np.sqrt(n))
    '''

    pickle.dump([x, approx], open('data/' + str(gi) + '.dicke', 'wb'))

if __name__ == '__main__':
    #gather_dicke()
    gather_classical()
