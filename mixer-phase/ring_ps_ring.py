import networkx as nx
import numpy as np
from scipy.special import comb
from math import pi
from scipy.optimize import basinhopping
import common

def qaoa(G, C, M, k, p, m, gamma, beta0, beta1):
    state = np.zeros(2**len(G.nodes))
    state = common.prep(state, G, k, m)
    state = common.mixer(state, M, beta0)
    for i in range(p):
        state = common.phase_separator(state, C, gamma)
        state = common.mixer(state, M, beta1)
    return state

def opt(args, *extras):
    state = qaoa(extras[0], extras[1], extras[2], extras[3], extras[4], extras[5], args[0], args[1], args[2])
    return -common.expectation(extras[0], state)

def ring_ps_ring(G, k, p, num_steps):
    C = common.create_C(G)
    M = common.create_ring_M(len(G.nodes))
    exp_arr = []
    angle_arr = []

    num_init = int(comb(len(G.nodes), k))
    for m in range(0, num_init):
        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p, m), 'bounds': [[0,pi/2],[0,pi],[0,pi]]}
        optimal = basinhopping(opt, np.array([pi/4, pi/2, pi/2]), minimizer_kwargs=kwargs, niter=num_steps)
        angle_arr.append(optimal.x)
        exp_arr.append(-optimal.fun)

    avg = 0
    for entry in exp_arr: avg += entry*(1/num_init)
    return avg, angle_arr

if __name__ == '__main__':
    G = nx.graph_atlas(6)
    #exp, angles = old_ring_ps_ring(G, int(len(G.nodes)/2), 1, 20)
    #print('exp: ' + str(exp))
