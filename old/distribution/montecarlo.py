import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import networkx as nx
import random
from math import pi
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
        if i % 10 == 0: print(i)
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

def main():
    max_p = 5
    s = 50
    n = 50
    z = 2.576
    gi = 91
    max_exp, max_std, error = [], [], []
    for p in range(1, max_p+1):
        print('p = ' + str(p))
        data = monte_carlo(gi, p, s, n)
        best = brute_force(gi)
        data = [x/best for x in data]

        max_exp.append(np.average(data))
        max_std.append(np.std(data))
        error.append(z*np.std(data)/np.sqrt(n))
        print(max_exp)
        print(max_std)
        print(error)

    x = [i for i in range(1, max_p+1)]
    plt.errorbar(x, max_exp, yerr=error, fmt='-o')
    plt.xlabel('p')
    plt.ylabel('Expectation value <C>')
    plt.title('Best solution via Monte Carlo sampling with \ns=' + str(s) + ' for gi=' + str(gi))
    plt.gca().set_ylim([0.8, 1])

    plt.show()

if __name__ == '__main__':
    main()
