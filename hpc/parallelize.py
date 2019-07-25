import time
import ctypes
import random
import multiprocessing
from multiprocessing import Value
import pdb

samples = 8
data = None

def f(i):
    global data
    print(i)
    for j in range(data[0]):
        k = 200**200**2
    return i

def par():
    with multiprocessing.Pool() as pool:
        pool.map(f, range(0, samples))

def seq():
    for i in range(0, samples):
        f(i)

if __name__ == '__main__':
    data = [i*i for i in range(10000)]
    data[0] = 10000000

    t1 = time.time()
    print('seq')
    seq()
    t2 = time.time()
    print('par')
    par()
    t3 = time.time()
    print('seq: ' + str(t2-t1))
    print('par: ' + str(t3-t2))
