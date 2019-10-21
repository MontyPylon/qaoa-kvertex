import numpy as np
import networkx as nx
from math import pi
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from scipy.optimize import show_options
import sys
sys.path.insert(0, '../common/')
import os
import common
import pickle
import random
import time
import datetime

def qaoa(gb, G, C, M , k, p):
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return -common.expectation(G, k, state)

def monte_best(G, C, M, k, p, num_samples):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    best_exp = -1
    for i in range(num_samples):
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        value = -qaoa(angles, G, C, M, k, p)
        if value > best_exp: best_exp = value
    return best_exp

def basin_best(G, C, M, k, p, num_samples):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
    evals = 0
    n_iter = 10
    max_fun = num_samples/10
    best_exp = -1
    while evals < num_samples:
        angles = [random.uniform(low_g, up_g) if j < p else random.uniform(low_b, up_b) for j in range(2*p)]
        kwargs = {'method': 'L-BFGS-B', 'options': {'maxfun': max_fun}, 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=n_iter, disp=True)
        evals += optimal.nfev
        if -optimal.fun > best_exp: best_exp = -optimal.fun
    return best_exp

def inter_best(G, C, M, k, p, num_samples):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]
    gamma = np.linspace(0.4, 1.7, p+2)
    gamma = np.delete(gamma, len(gamma)-1)
    gamma = np.delete(gamma, 0)
    gamma = list(gamma)
    beta = np.linspace(0.18, 0, p+2)
    beta = np.delete(beta, len(beta)-1)
    beta = np.delete(beta, 0)
    beta = list(beta)
    orig = gamma.copy()
    orig.extend(beta)

    evals = 0
    n_iter = 10
    max_fun = num_samples/10
    best_exp = -1
    while evals < num_samples:
        angles = []
        for j in range(2*p):
            offset = 0
            if j < p:
                offset = random.uniform(-0.3,0.3)
            else:
                offset = random.uniform(-0.05, 0.05)
            angles.append(orig[j] + offset)
            if angles[j] < 0: angles[j] = 0
        kwargs = {'method': 'L-BFGS-B', 'options': {'maxfun': max_fun}, 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=n_iter, disp=True)
        evals += optimal.nfev
        if -optimal.fun > best_exp: best_exp = -optimal.fun
    return best_exp

def inter_best2(G, C, M, k, p, num_samples):
    low_g, up_g = 0, 2*pi
    low_b, up_b = 0, pi/2
    bounds = [[low_g,up_g] if j < p else [low_b,up_b] for j in range(2*p)]

    evals = 0
    n_iter = 10
    max_fun = num_samples/10
    best_exp = -1
    while evals < num_samples:
        left_g = np.random.uniform(0, 1)
        right_g = np.random.uniform(1, 2)
        gamma = np.linspace(left_g, right_g, p)
        left_b = np.random.uniform(0.1, 0.2)
        right_b = np.random.uniform(0, 0.1)
        beta = np.linspace(left_b, right_b, p)
        angles = gamma.copy()
        angles.extend(beta)

        kwargs = {'method': 'L-BFGS-B', 'options': {'maxfun': max_fun}, 'args': (G, C, M, k, p), 'bounds': bounds}
        optimal = basinhopping(qaoa, angles, minimizer_kwargs=kwargs, niter=n_iter, disp=True)
        evals += optimal.nfev
        if -optimal.fun > best_exp: best_exp = -optimal.fun
    return best_exp
