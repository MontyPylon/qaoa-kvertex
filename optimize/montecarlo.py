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
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def neldermead(G, C, M, k, p, angles, eps):
    options = {'disp': False, 'maxfev': 100, 'fatol': eps, 'adaptive': True}
    optimal = minimize(qaoa, angles, method='Nelder-Mead', args=(G, C, M, k, p), options=options)
    return optimal

def lbfgsb(G, C, M, k, p, angles, eps, bounds):
    options = {'disp': None, 'maxfun': 100, 'gtol': eps, 'ftol': eps}
    optimal = minimize(qaoa, angles, method='L-BFGS-B', bounds=bounds, args=(G, C, M, k, p), options=options)
    return optimal

def work(gi, p, s):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    G, C, M, k = common.get_complete(gi)
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
    eps = 1e-3
    best_exp, best_angles = -1, []
    total_eval = 0
    while(total_eval < s):
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]

        #optimal = neldermead(G, C, M, k, p, angles, eps)
        optimal = lbfgsb(G, C, M, k, p, angles, eps, bounds)

        total_eval += optimal.nfev
        if -optimal.fun > best_exp + eps:
            best_exp = -optimal.fun
            best_angles = [list(optimal.x)]
        elif -optimal.fun > best_exp - eps:
            best_angles.append(list(optimal.x))
    return best_exp, best_angles

def basin(gi, p, s):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    G, C, M, k = common.get_complete(gi)
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
    eps = 1e-3
    total_eval = 0
    best_exp, best_angles = -1, []
    while total_eval < s:
        angles = [0 for _ in range(2*p)]
        opt = {'disp': None, 'gtol': eps, 'ftol': eps}
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds, 'options': opt}
        optimal = basinhopping(qaoa, angles, niter=200, minimizer_kwargs=kwargs, disp=False)
        total_eval += optimal.nfev
        print(total_eval)
        if -optimal.fun > best_exp + eps:
            best_exp = -optimal.fun
            best_angles = [list(optimal.x)]
        elif -optimal.fun > best_exp - eps:
            best_angles.append(list(optimal.x))
    return best_exp, best_angles

if __name__ == '__main__':
    gi = 404
    p = 1
    s = 100
    for p in range(1,20):
        rank_exp, rank_angles = basin(gi, p, s*(3**p))
        print('best_exp: ' + str(rank_exp))

