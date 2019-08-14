import numpy as np
from math import pi
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys
sys.path.insert(0, '../common/')
import os
import common
import pickle
import random
import time

def qaoa(gb, G, C, M , k, p):
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def next_level(G, C, M, k, p, prev_exp):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    best_exp = -1
    total_eval = 0
    while(best_exp < prev_exp):
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        value = -qaoa(angles, G, C, M, k, p)
        if value > best_exp: best_exp = value
        total_eval += 1
    return best_exp, total_eval

if __name__ == '__main__':
    gi = 400
    G, C, M, k = common.get_complete(gi)
    s = 1000
    exp = [0]*s
    prev_best = 0
    samples = []
    error = []

    for p in range(1,10):
        sample_vec = []
        for i in range(s):
            new_exp, num_samples = next_level(G, C, M, k, p, prev_best)
            exp[i] = new_exp
            sample_vec.append(num_samples)
            if i % 50 == 0: print('\t i: ' + str(i) + ', avg: ' + str(np.average(sample_vec)))
        prev_best = np.average(exp)
        print('new-avg: ' + str(prev_best))
        samples.append(np.average(sample_vec))
        error.append(2.576*np.std(sample_vec)/np.sqrt(s))
        print('samples: ' + str(np.average(sample_vec)) + '\t std: ' + str(np.std(sample_vec)) + '\t error: ' + str(2.576*np.std(sample_vec)/np.sqrt(s)))
        pickle.dump([samples, error, ], open('data/samples', 'wb'))
