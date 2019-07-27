import networkx as nx
import numpy as np
from scipy.optimize import basinhopping
from math import pi
import random
import common

def qaoa(a, gb):
    G, C, M, k, p = a[0], a[1], a[2], a[3], a[4]
    state = common.dicke(len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, gb[i])
        state = common.mixer(state, M, gb[p+i])
    return state

def qaoa_restricted(a, opt):
    G, C, M, k, p, prev = a[0], a[1], a[2], a[3], a[4], a[5]
    state = np.zeros(2**len(G.nodes))
    state = common.dicke(state, len(G.nodes), k)
    for i in range(p):
        if i != p - 1:
            state = common.phase_separator(state, C, prev[0][i])
            state = common.mixer(state, M, prev[1][i])
        else:
            state = common.phase_separator(state, C, opt[0])
            state = common.mixer(state, M, opt[1])
    return state

def opt(args, *a):
    state = qaoa(a, args)
    return -common.expectation(a[0], state)

def opt_restricted(args, *a):
    state = qaoa_restricted(a, args)
    return -common.expectation(a[0], state)

def main(G, k, p, num_steps):
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    init = [[0,pi/2] if i < p else [0,pi] for i in range(2*p)]
    guess = [pi/4 if i < p else pi/2 for i in range(2*p)]
    kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': init}
    optimal = basinhopping(opt, guess, minimizer_kwargs=kwargs, niter=num_steps, disp=True)
    return -optimal.fun, optimal.x

def LHS(p, num_samples):
    g = [(pi/(2*num_samples))*(i + 0.5) for i in range(num_samples)]
    b = [(pi/num_samples)*(i + 0.5) for i in range(num_samples)]
    valid_g = [[i for i in range(num_samples)] for j in range(p)]
    valid_b = [[i for i in range(num_samples)] for j in range(p)]
    sample_g = [[] for j in range(num_samples)]
    sample_b = [[] for j in range(num_samples)]
    for j in range(p):
        for i in range(num_samples):
            vg = random.randint(0, len(valid_g[j])-1)
            vb = random.randint(0, len(valid_b[j])-1)
            sample_g[i].append(g[valid_g[j][vg]])
            sample_b[i].append(b[valid_b[j][vb]])
            del valid_g[j][vg]
            del valid_b[j][vb]
    return sample_g, sample_b

def MLHS(G, k, p, num_samples):
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    num_steps = 2
    init = [[0,pi/2] if i < p else [0,pi] for i in range(2*p)]
    sample_g, sample_b = LHS(p, num_samples)
    exp, angles = -1, []
    for i in range(num_samples):
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': init}
        optimal = basinhopping(opt, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=num_steps, disp=True)
        if -optimal.fun > exp:
            exp = -optimal.fun
            angles = optimal.x
    return exp, angles

def restricted_MLHS(G, k, p, num_samples, prev):
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    num_steps = 2
    init = [[0,pi/2], [0,pi]]
    sample_g, sample_b = LHS(1, num_samples)
    exp, angles = -1, []
    for i in range(num_samples):
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p, prev), 'bounds': init}
        optimal = basinhopping(opt_restricted, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=num_steps, disp=True)
        if -optimal.fun > exp:
            exp = -optimal.fun
            angles = optimal.x
    return exp, angles

if __name__ == '__main__':
    gi = 6
    G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
    exp, angles = MLHS(G, int(len(G.nodes)/2), 3, 5)
    print('exp: ' + str(exp))
    print('angles: ' + str(angles))
