import numpy as np
from math import pi
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from scipy.special import comb
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

def qaoa_initial(gb, *a):
    G, C, M, k, p, m = a[0], a[1], a[2], a[3], a[4], a[5]
    size = int(comb(len(G.nodes), k))
    state = np.zeros(size)
    state[m] = 1
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def initial_ring(gi):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    G, C, M, k = common.get_ring(gi)
    eps = 1e-3
    avg = []
    error = []
    size = int(comb(len(G.nodes), k))
    for p in range(1, 7):
        exp, e = [], 0
        bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
        for i in range(size):
            angles = [random.uniform(0,2*pi) if j < p else random.uniform(0,pi/2) for j in range(2*p)]
            opt = {'disp': None, 'gtol': eps, 'ftol': eps}
            kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p, i), 'bounds': bounds, 'options': opt}
            optimal = basinhopping(qaoa_initial, angles, niter=50, minimizer_kwargs=kwargs, disp=False)
            print('p: ' + str(p) + '\t' + str(-optimal.fun) + '\t' + str(optimal.x))
            exp.append(-optimal.fun/9)
        avg.append(np.average(exp))
        error.append(2.576*np.std(exp)/np.sqrt(size))
        print('avg: ' + str(avg) + '\t' + str(error))
        pickle.dump([avg, error], open('data/91.initial_ring', 'wb'))


def initial_complete(gi):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    G, C, M, k = common.get_complete(gi)
    eps = 1e-3
    avg = []
    error = []
    size = int(comb(len(G.nodes), k))
    print(size)
    for p in range(1, 7):
        exp, e = [], 0
        bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
        for i in range(size):
            angles = [random.uniform(0,2*pi) if j < p else random.uniform(0,pi/2) for j in range(2*p)]
            opt = {'disp': None, 'gtol': eps, 'ftol': eps}
            kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p, i), 'bounds': bounds, 'options': opt}
            optimal = basinhopping(qaoa_initial, angles, niter=50, minimizer_kwargs=kwargs, disp=False)
            print('i: ' + str(i) + ', p: ' + str(p) + '\t' + str(-optimal.fun) + '\t' + str(optimal.x))
            exp.append(-optimal.fun/9)
        avg.append(np.average(exp))
        error.append(2.576*np.std(exp)/np.sqrt(size))
        print('avg: ' + str(avg) + '\t' + str(error))
        pickle.dump([avg, error], open('data/91.initial_complete', 'wb'))

def dicke_ring(gi):
    low_g, up_g = 0, 2
    low_b, up_b = 0, 0.4
    G, C, M, k = common.get_ring(gi)
    eps = 1e-3
    exp = []
    for p in range(1, 20):
        sample_g = np.linspace(up_g, low_g, num=p+2)
        sample_g = sample_g[1:p+1]
        sample_b = np.linspace(low_b, up_b, num=p+2)
        sample_b = sample_b[1:p+1]
        angles = np.append(sample_g, sample_b)
        bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
        opt = {'disp': None, 'gtol': eps, 'ftol': eps}
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds, 'options': opt}
        optimal = basinhopping(qaoa, angles, niter=300, minimizer_kwargs=kwargs, disp=False)
        print('p: ' + str(p) + '\t' + str(-optimal.fun) + '\t' + str(optimal.x))
        exp.append(-optimal.fun/9)
        pickle.dump(exp, open('data/91.dicke_ring', 'wb'))

def dicke_complete(gi):
    low_g, up_g = 0, 2
    low_b, up_b = 0, 0.2
    G, C, M, k = common.get_complete(gi)
    eps = 1e-3
    exp = []
    for p in range(1, 20):
        sample_g = np.linspace(up_g, low_g, num=p+2)
        sample_g = sample_g[1:p+1]
        sample_b = np.linspace(low_b, up_b, num=p+2)
        sample_b = sample_b[1:p+1]
        angles = np.append(sample_g, sample_b)
        bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
        opt = {'disp': None, 'gtol': eps, 'ftol': eps}
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': bounds, 'options': opt}
        optimal = basinhopping(qaoa, angles, niter=300, minimizer_kwargs=kwargs, disp=False)
        print('p: ' + str(p) + '\t' + str(-optimal.fun) + '\t' + str(optimal.x))
        exp.append(-optimal.fun/9)
        pickle.dump(exp, open('data/91.dicke_complete', 'wb'))

if __name__ == '__main__':
    gi = 91
    #dicke_complete(gi)
    #dicke_ring(gi)
    #initial_complete(gi)
    #initial_ring(gi)
    rho_complete(gi)
    print('edit to not overwrite other data')
