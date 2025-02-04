import numpy as np
import sys
sys.path.insert(0, '/home/montypylon/lanl/qaoa-kvertex/mixer-phase')
import common
from scipy.linalg import expm
from math import pi

def check(n, beta):
    print('creating mixer')
    eps = 0.001
    M = common.create_complete_M(n)
    print('exponentiating mixer')
    ring = expm(np.complex(0,-1)*beta*pi*M)
    total = 0

    print('checking mixer')
    max = 2**n
    # check if all off diagonal elements are 0
    for i in range(max):
        for j in range(max):
            if i != j:
                total += np.real(ring[i,j]*np.conj(ring[i,j]))
                if total > eps:
                    return False
    # Now check phases are equal
    noteq = False
    for i in range(1, max):
        if (np.real(ring[0,0]) - np.real(ring[i,i])) > eps or (np.imag(ring[0,0]) - np.imag(ring[i,i])) > eps:
            return False

    if total < 1e-12: print('******************************')
    print('b: ' + str(beta) + '\t t: ' + str(total) + '\t t*pi: ' + str(beta*pi))
    return True

def main():
    n = 10
    #k = 1
    eps = 0.001
    start = 0.99
    end = 1.01
    M = common.create_complete_M(n)
    period = []
    step = 0.01
    beta = start
    while beta < end:
        ring = expm(np.complex(0,-1)*beta*pi*M)
        total = 0
        # check if all off diagonal elements are 0
        for i in range(2**n):
            for j in range(2**n):
                #if common.num_ones(j) == k:
                if i != j:
                    total += np.real(ring[i,j]*np.conj(ring[i,j]))
        if total < eps:
            # Now check phases are equal
            noteq = False
            for i in range(1, 2**n):
                if (np.real(ring[0,0]) - np.real(ring[i,i])) > eps or (np.imag(ring[0,0]) - np.imag(ring[i,i])) > eps:
                    noteq = True
                    break
            if noteq:
                beta += step
                continue
            if total < 1e-12: print('******************************')
            print('b: ' + str(beta) + '\t t: ' + str(total) + '\t t*pi: ' + str(beta*pi))
            period.append(beta)
        beta += step

    print(period)
    dist = []
    for i in range(len(period)-1):
        dist.append(period[i+1] - period[i])
    print(dist)

def test():
    n = 4
    eps = 1e-15
    M = common.create_ring_M(n)
    beta = 17.5*pi
    ring = expm(np.complex(0,-1)*beta*M) - np.eye(2**n)
    total = 0
    for i in range(2**n):
        for j in range(2**n):
            ring[i,j] = np.real(ring[i,j]*np.conj(ring[i,j]))
            if ring[i,j] < eps: ring[i,j] = 0
            total += ring[i,j]
    print(np.real(total))



if __name__ == '__main__':
    for j in range(2,20):
        print('----------------')
        print('j=' + str(j) + ', ' + str(check(j, 1)))
        print('----------------')
    #main()
    #half()
    #test()
