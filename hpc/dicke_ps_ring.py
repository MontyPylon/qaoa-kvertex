import networkx as nx
import numpy as np
from scipy.optimize import basinhopping
from math import pi
import common

def qaoa(G, C, M, k, p, gamma, beta):
    state = np.zeros(2**len(G.nodes))
    state = common.dicke(state, G, k)
    for i in range(p):
        state = common.phase_separator(state, C, gamma)
        state = common.ring_mixer(state, M, beta)
    return state

def opt(args, *a):
    state = qaoa(a[0], a[1], a[2], a[3], a[4], args[0], args[1])
    return -common.expectation(a[0], state)

def dicke_ps_ring(G, k, p, num_steps):
    C = common.create_C(G)
    M = common.create_M(G)
    kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': [[0,pi/2],[0,pi]]}
    optimal = basinhopping(opt, np.array([pi/4, pi/2]), minimizer_kwargs=kwargs, niter=num_steps)
    return -optimal.fun, optimal.x

if __name__ == '__main__':
    G = nx.graph_atlas(6)
    exp, angles = dicke_ps_ring(G, int(len(G.nodes)/2), 1, 20)
    print('exp: ' + str(exp))
