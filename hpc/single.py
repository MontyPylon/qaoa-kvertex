import time
import threading

def t():
    with open('/dev/urandom', 'rb') as f:
        for x in range(10000):
            f.read(4 * 65535)

if __name__ == '__main__':
    start_time = time.time()
    t()
    t()
    t()
    t()
    print("Sequential run time: %.2f seconds" % (time.time() - start_time))

    start_time = time.time()
    t1 = threading.Thread(target=t)
    t2 = threading.Thread(target=t)
    t3 = threading.Thread(target=t)
    t4 = threading.Thread(target=t)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    print("Parallel run time: %.2f seconds" % (time.time() - start_time))
'''
from math import pi
import numpy as np
from scipy.linalg import expm
import time

n = 5
samples = 2**9
num_cores = 8

t1 = time.time()
for i in range(num_cores):
    np.random.seed(1)
    for j in range(samples):
        A = np.random.rand(2**n, 2**n)
        B = expm(np.complex(0,1)*A)
t2 = time.time()
print('seq ' + ' took ' + str(t2-t1))
'''
