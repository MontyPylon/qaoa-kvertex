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

def qaoa(gb, *a):
    G, C, M, k, p = a[0], a[1], a[2], a[3], a[4]
    state = common.dicke(len(G.nodes))
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, state)

def work(gi, p, s):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    G, C, M, k = common.get_complete(gi)
    sample_g, sample_b = common.MLHS(p, s, low_g, up_g, low_b, up_b)
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
    eps = 1e-3
    best_exp, best_angles = -1, []
    total_eval = 0
    for i in range(s):
        angles = sample_g[i]
        angles.extend(sample_b[i])
        options = {'disp': False, 'maxfev': 100, 'fatol': eps, 'adaptive': True}
        optimal = minimize(qaoa, angles, method='Nelder-Mead', args=(G, C, M, k, p), options=options)
        total_eval += optimal.nfev
        print('total_eval: ' + str(total_eval))
        if -optimal.fun > best_exp + eps:
            best_exp = -optimal.fun
            best_angles = [list(optimal.x)]
            print('s: ' + str(i) + ', ' + str(-optimal.fun) + ', angles: ' + str(optimal.x))
        elif -optimal.fun > best_exp - eps:
            best_angles.append(list(optimal.x))
    #print(str(rank) + ', ' + str(best) + ', ' + str(angles))
    ''' 
    G, C, M, k = common.get_complete(gi)
    options = {'disp': 1, 'maxfun': 100, 'gtol': 1e-2}
    optimal = minimize(qaoa, np.array([0.5*pi, 0.1*pi]), method='L-BFGS-B', bounds=[[0,2*pi],[0,pi/2]], args=(G, C, M, k, p), options=options)
    print('optimal: ' + str(-optimal.fun))
    print('angles: ' + str(optimal.x))
    return -optimal.fun, optimal.x
    '''
    print('')
    print(best_exp)
    print(best_angles)
    return best_exp, best_angles

if __name__ == '__main__':
    gi = 91
    p = 1
    s = 10
    rank_exp, rank_angles = work(gi, p, s)

