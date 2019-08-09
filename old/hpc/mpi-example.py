from mpi4py import MPI
from math import pi
import numpy as np
from scipy.linalg import expm
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 7
samples = 2**3
num_cores = 8

if size == 1:
    t1 = time.time()
    for i in range(num_cores):
        np.random.seed(1)
        for j in range(samples):
            A = np.random.rand(2**n, 2**n)
            B = expm(np.complex(0,1)*A)
    t2 = time.time()
    print('seq ' + ' took ' + str(t2-t1))
else:
    t1 = time.time()
    np.random.seed(rank)
    for j in range(samples):
        A = np.random.rand(2**n, 2**n)
        B = expm(np.complex(0,1)*A)
    t2 = time.time()
    print(str(t2-t1))
