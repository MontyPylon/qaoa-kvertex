import datetime
import os
import pickle
import networkx as nx
# from dicke_hprime_parityring import dicke_hprime_parityring
# from pring_ps_pring import pring_ps_pring
from brute_force import brute_force
from dicke_ps_ring import dicke_ps_ring
from ring_ps_ring import ring_ps_ring

def compare():
    # min: 1, max: 995
    start = 1
    end = 3
    p = 1
    n = 3
    dict_exp1 = {}
    dict_exp2 = {}
    dict_exp3 = {}
    dict_angles1 = {}
    dict_angles2 = {}
    dict_angles3 = {}
    
    for gi in range(start, end+1):
        print(str(gi) + '/' + str(end) + '\t' + str(datetime.datetime.now().time()))
        G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
        k = int(len(G.nodes)/2)
        if k == 0: k = 1
        key = tuple([gi, k])

        exp, angles = brute_force(G, k, p, n)
        dict_exp1[key] = exp
        dict_angles1[key] = angles

        exp, angles = dicke_ps_ring(G, k, p, n)
        dict_exp2[key] = exp
        dict_angles2[key] = angles

        exp, angles = ring_ps_ring(G, k, p, n)
        dict_exp3[key] = exp
        dict_angles3[key] = angles

    print('Dumping pickles at: ' + str(datetime.datetime.now().time()))
    pickle.dump(dict_exp1, open('data/brute_force.exp', 'wb'))
    pickle.dump(dict_exp2, open('data/dicke_ps_ring.exp', 'wb'))
    pickle.dump(dict_exp3, open('data/ring_ps_ring.exp', 'wb'))
    pickle.dump(dict_angles1, open('data/brute_force.angles', 'wb'))
    pickle.dump(dict_angles2, open('data/dicke_ps_ring.angles', 'wb'))
    pickle.dump(dict_angles3, open('data/ring_ps_ring.angles', 'wb'))

    print('Finished at: ' + str(datetime.datetime.now().time()))
    print('dict_exp1:\n' + str(dict_exp1))
    print('dict_exp2:\n' + str(dict_exp2))
    print('dict_exp3:\n' + str(dict_exp3))
    print('dict_angles1:\n' + str(dict_angles1))
    print('dict_angles2:\n' + str(dict_angles2))
    print('dict_angles3:\n' + str(dict_angles3))

if __name__ == '__main__':
    compare()
