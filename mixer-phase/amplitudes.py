import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.special import comb
import common
import random

def get_exp(G, gi, k, p, num_steps, method, method_string):
    exp = None
    found = False
    exp_dict = {}
    path = 'data/' + method_string + '.exp'

    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                exp_dict = pickle.load(f)
                key = tuple([gi, k])
                if key in exp_dict:
                    found = True
                    exp = exp_dict[key]
            except Exception as e:
                print('Error opening dictionary for ' + method_string)
                print(e)

    if not found or exp is None:
        exp, angles = method(G, k, p, num_steps)
        key = tuple([gi, k])
        exp_dict[key] = exp
        pickle.dump(exp_dict, open(path, 'wb'))

    return exp

def prep(n, k):
    state = np.zeros(2**n)
    for i in range(0, 2**n):
        if common.num_ones(i) == k:
            state[i] = 1
            break
    return state

def dicke(n, k):
    state = np.zeros(2**n)
    for i in range(0, 2**n):
        if common.num_ones(i) == k:
            state[i] = 1/np.sqrt(comb(n, k))
    return state

def f(t, n, k):
    t = t*pi
    state = prep(n, k)
    #state = dicke(n, k)
    state = common.mixer(state, common.create_ring_M(n), t)
    state = [np.real(np.conj(s)*s) for s in state]
    probs = []
    for i in range(0, 2**n):
        if common.num_ones(i) == k:
            probs.append(state[i])
    return probs

def amplitudes():
    n = 2
    k = 1
    shell = [[] for x in range(int(comb(n, k)))]
    t = np.arange(0, 1, 0.005)

    for i in t:
        out = f(i, n, k)
        for s in range(0, len(shell)):
            shell[s].append(out[s])

    for s in range(len(shell)):
        #plt.plot(t, shell[s], label=str(s), linewidth=int(comb(n, k))+1-s, zorder=5*s)
        plt.plot(t, shell[s], label=str(s))
    #plt.xticks(np.arange(start-1, end, step=1))

    plt.legend()
    plt.xlabel('t/pi')
    plt.ylabel('Probability')
    plt.title('Ring mixer, n=' + str(n) + ', k=' + str(k))
    plt.show()
        
if __name__ == '__main__':
    #generate_graphs()
    amplitudes()
