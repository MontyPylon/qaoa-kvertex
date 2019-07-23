import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pickle
import random
import os

import sys
sys.path.insert(0, '/home/montypylon/lanl/qaoa-kvertex/mixer-phase')
import common
import dicke_ps_complete

G, C, M, k = None, None, None, None

def exp_angles(G, gi, k, p, num_steps, method, method_string):
    angles = None
    found = False
    angle_dict = {}
    folder = 'angles/'
    path = folder + 'optimal.exp'

    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                angle_dict = pickle.load(f)
                key = tuple([gi, p])
                if key in angle_dict:
                    found = True
                    angles = angle_dict[key]
            except Exception as e:
                print('Error opening dictionary for ' + method_string)
                print(e)

    if not found:
        exp, angles = method(G, k, p, num_steps)
        key = tuple([gi, p])
        angle_dict[key] = angles
        if not os.path.exists(folder): os.mkdir(folder)
        pickle.dump(angle_dict, open(path, 'wb'))
    return angles

#def top(G, gi, k, p, num_steps, method, method_string):


if __name__ == '__main__':
    gi = 91
    G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    num_steps = 25

    angles = []
    max_p = 1
    for p in range(1, max_p+1):
        angles.append(exp_angles(G, gi, k, p, num_steps, dicke_ps_complete.main, 'dicke_ps_complete'))

    '''
    x = [i+1 for i in range(p)]
    g = [angles[i] for i in range(p)]
    plt.plot(x, g, '-o', label=str(gi))
    #b = [angles[i+p] for i in range(p)]
    #plt.plot(x, b, '-o', label='beta')

    axes = plt.gca()
    axes.set_ylim([0, pi])
    plt.xticks(np.arange(1, p+1, step=1))
    plt.legend()
    plt.xlabel('gamma_i')
    plt.ylabel('value of gamma_i')
    plt.title('Gamma & beta vector correlation \ngi=' + str(gi) + \
              ', p=' + str(p) + ', k=floor(n/2), iter=' + str(num_steps))
    plt.show()
    '''
