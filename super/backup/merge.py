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


def merge():
    exp_dict1 = {}
    exp_dict2 = {}
    path1 = 'merge/dicke_ps_ring.angles'
    path2 = 'merge/data4/dicke_ps_ring.angles'

    if os.path.exists(path1):
        with open(path1, 'rb') as f:
            try:
                exp_dict1 = pickle.load(f)
            except: 
                print('Error opening dictionary for ' + path1)
    
    if os.path.exists(path2):
        with open(path2, 'rb') as f:
            try:
                exp_dict2 = pickle.load(f)
            except: 
                print('Error opening dictionary for ' + path2)

    exp_dict1.update(exp_dict2) 

    pickle.dump(exp_dict1, open('merge/dicke_ps_ring.angles', 'wb'))
        
if __name__ == '__main__':
    merge()
