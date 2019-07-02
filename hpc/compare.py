import datetime
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
# from dicke_hprime_parityring import dicke_hprime_parityring
# from pring_ps_pring import pring_ps_pring
from brute_force import brute_force
from dicke_ps_ring import dicke_ps_ring
from ring_ps_ring import ring_ps_ring
from old_ring_ps_ring import old_ring_ps_ring


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
            except: 
                print('Error opening dictionary for ' + method_string)
       
    if not found or exp is None:
        exp, angle = method(G, k, p, num_steps)
        key = tuple([gi, k])
        exp_dict[key] = exp
        pickle.dump(exp_dict, open(path, 'wb'))

    return exp

def compare():
    # min: 1, max: 995
    start = 1
    end = 10
    p = 1
    n = 3
    x = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    
    for gi in range(start, end+1):
        print(str(gi) + '/' + str(end) + '\t' + str(datetime.datetime.now().time()))
        G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
        k = int(len(G.nodes)/2)
        if k == 0:
            k = 1
        #k = 4
        #x.append('gi=' + str(gi) + ',n=' + str(len(G.nodes)) + ',k=' + str(k))
        x.append(gi)
        #exps = write_angles(gi)

        y1.append(get_exp(G, gi, k, p, n, brute_force, 'brute_force'))
        y2.append(get_exp(G, gi, k, p, n, dicke_ps_ring, 'dicke_ps_ring'))
        y3.append(get_exp(G, gi, k, p, n, ring_ps_ring, 'ring_ps_ring'))
        y4.append(get_exp(G, gi, k, p, n, old_ring_ps_ring, 'old_ring_ps_ring'))

    plt.plot(x, y1, '-bo', label='optimal')
    plt.plot(x, y2, '-go', label='dicke_ps_ring')
    plt.plot(x, y3, '-ro', label='ring_ps_ring')
    plt.plot(x, y4, '-yo', label='old_ring_ps_ring')

    plt.legend()

    plt.xlabel('Graph atlas index')
    plt.ylabel('<C>: expectation of number of edges touched')
    plt.title('<C> vs. graph atlas\np=' + str(p) + ', k=floor(n/2), grid_size=' + str(n) + 'x' + str(n))
    plt.show()
        
if __name__ == '__main__':
    #generate_graphs()
    compare()
