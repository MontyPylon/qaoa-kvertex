import datetime
import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from brute_force import brute_force
from dicke_ps_ring import dicke_ps_ring
from ring_ps_ring import ring_ps_ring
import dicke_ps_complete

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
        print('method: ' + method_string)
        print('angles: ' + str(angles))
        key = tuple([gi, k])
        exp_dict[key] = exp
        pickle.dump(exp_dict, open(path, 'wb'))

    return exp

def compare():
    # min: 1, max: 995
    start = 1
    end = 20
    p = 1
    num_steps = 25
    x, y1, y2, y3, y4, y5 = [], [], [], [], [], []

    for gi in range(start, end+1):
        print(str(gi) + '/' + str(end) + '\t' + str(datetime.datetime.now().time()))
        G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
        k = int(len(G.nodes)/2)
        if k == 0:
            k = 1
        x.append(str(gi) + ',' + str(len(G.nodes)) + ',' + str(k))
        #x.append(gi)
        y1.append(get_exp(G, gi, k, p, num_steps, brute_force, 'brute_force'))
        y2.append(get_exp(G, gi, k, p, num_steps, dicke_ps_ring, 'dicke_ps_ring'))
        y3.append(get_exp(G, gi, k, p, num_steps, ring_ps_ring, 'ring_ps_ring'))
        y4.append(get_exp(G, gi, k, p, num_steps, dicke_ps_complete, 'dicke_ps_complete'))


    for i in range(0, len(y1)):
        y2[i] /= y1[i]
        y3[i] /= y1[i]
        y4[i] /= y1[i]
        y1[i] = 1

    plt.plot(x, y1, '-', label='optimal')
    plt.plot(x, y2, '-o', label='dicke_ps_ring')
    plt.plot(x, y3, '-o', label='ring_ps_ring')
    plt.plot(x, y4, '-o', label='dicke_ps_complete')

    #plt.xticks(np.arange(start-1, end, step=1))

    plt.legend()
    plt.xlabel('Graph atlas index, # of nodes, k')
    plt.ylabel('Approximation ratio')
    plt.title('Approximation ratio vs. graph atlas\np=' + str(p) + ', k=floor(n/2), iter=' + str(num_steps))
    plt.show()
        
if __name__ == '__main__':
    #generate_graphs()
    compare()
