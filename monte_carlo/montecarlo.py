import matplotlib.pyplot as plt
import pickle
from itertools import combinations
import numpy as np
import networkx as nx
import random
from math import pi
import datetime
import os
import sys
sys.path.insert(0, '/home/montypylon/lanl/qaoa-kvertex/mixer-phase/')
import common
import dicke_ps_complete

def monte_carlo(gi, p, s, num_samples):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    data = []
    for i in range(num_samples):
        max_exp = -1
        for j in range(s):
            angles = [random.uniform(0,pi/2) if i < p else random.uniform(0,pi) for i in range(2*p)]
            exp = common.expectation(G, dicke_ps_complete.qaoa([G, C, M, k, p], angles))
            if exp > max_exp: max_exp = exp
        data.append(max_exp)
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

def gather_data():
    max_p = 20
    s = 50
    n = 50
    x = [i for i in range(1, max_p+1)]
    z = 2.576 # z* for 99% confidence interval

    for i in range(1, 1000):
        gi = random.randint(31, 995)
        print('i, gi = ' + str(i) + ', ' + str(gi) + '\t' + str(datetime.datetime.now().time()))

        max_exp, max_std, error = [], [], []
        for p in range(1, max_p+1):
            print('\tp = ' + str(p) + '\t' + str(datetime.datetime.now().time()))
            data = monte_carlo(gi, p, s, n)
            best = brute_force(gi)
            data = [x/best for x in data]
            max_exp.append(np.average(data))
            max_std.append(np.std(data))
            error.append(z*np.std(data)/np.sqrt(n))

        pickle.dump([x, max_exp, max_std, error, gi, s, n], open('data/' + str(gi) + '.carlo', 'wb'))


if __name__ == '__main__':
    gather_data()
