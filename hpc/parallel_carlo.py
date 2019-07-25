import numpy as np
import networkx as nx
import random
from math import pi
import os
import sys
sys.path.insert(0, '/home/montypylon/lanl/qaoa-kvertex/mixer-phase/')
import common
import time
from scipy.linalg import expm
import multiprocessing as mp

from mpi4py import MPI

# Global variables
#G = nx.read_gpickle('../benchmarks/atlas/91.gpickle')
#M = common.create_complete_M(len(G.nodes))

def multi(i):
    np.random.seed(i)
    expm((pi/(i+1))*np.random.rand(64, 64))
    return 0

if __name__ == '__main__':
    num = os.cpu_count()

    t1 = time.time()

    for i in range(num):
        multi(num)

    t2 = time.time()
    print('seq:\t' + str(t2-t1))

    with mp.Pool() as pool:
        results = pool.map(multi, range(num))

    t3 = time.time()
    print('pool:\t' + str(t3-t2))

    p = None
    for i in range(num):
        p = mp.Process(target=multi, args=(i,))
        p.start()
    p.join()

    t4 = time.time()
    print('proc:\t' + str(t4-t3))
