import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import common
import pickle
import datetime
import random
import os
import dicke_ps_complete

G, C, M, k = None, None, None, None


def exp_angles(G, gi, k, p, num_steps, method, method_string):
    exp = None
    angles = None
    found = False
    exp_dict = {}
    folder = 'angles/' + str(gi) + '/'
    path = folder + method_string + '.exp'

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
        print('angles: ' + str(angles))
        key = tuple([gi, k])
        exp_dict[key] = exp
        if not os.path.exists(folder): os.mkdir(folder)
        pickle.dump(exp_dict, open(path, 'wb'))
    return exp, angles

if __name__ == '__main__':
    gi = random.randint(1,995)
    print('gi = ' + str(gi))
    G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    num_steps = 20

    p = 3
    #for p in range(1, max_p+1):
    exp, angles = exp_angles(G, gi, k, p, num_steps, dicke_ps_complete.main, 'dicke_ps_complete')

    x = [i for i in range(len(angles))]
    plt.plot(x, angles[0], '-o')
    plt.plot(x, angles[1], '-o')


