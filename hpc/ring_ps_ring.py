import networkx as nx
import numpy as np
from scipy.misc import comb
from scipy.optimize import minimize
from math import pi
import datetime
from scipy.linalg import expm
import random
# Global variables
I = np.matrix('1 0; 0 1')
X = np.matrix('0 1; 1 0')
Y = np.matrix('0 -1; 1 0')*complex(0,1)
Z = np.matrix('1 0; 0 -1')
paulis = [I,X,Y,Z]

# take the kronecker (tensor) product of a list of matrices
def kron(m):
    if len(m) == 1:
        return m
    total = np.kron(m[0], m[1])
    if len(m) > 2:
        for i in range(2, len(m)):
            total = np.kron(total, m[i])
    return total

def num_ones(n):
    c = 0
    while n:
        c += 1
        n &= n - 1
    return c

def expectation(G, rho):
    init = [I]*len(G.nodes)
    total = complex(1,0)*np.zeros([2**len(G.nodes), 2**len(G.nodes)])
    for i in range(0, len(G.nodes)):
        init[i] = Z
        total -= G.degree[len(G.nodes) - 1 - i]*kron(init)
        # reset
        init = [I]*len(G.nodes)

    init = [I]*len(G.nodes)
    for edge in G.edges:
        init[len(G.nodes) - 1 - edge[0]] = Z
        init[len(G.nodes) - 1 - edge[1]] = Z
        total -= kron(init)
        # reset
        init = [I]*len(G.nodes)

    total += 3*np.identity(2**len(G.nodes))
    total = total/4

    final = np.trace(np.asmatrix(total)*np.asmatrix(rho))
    return np.real(final)

def phase_separator(rho, G, gamma):
    init = [I]*len(G.nodes)
    total = complex(1,0)*np.zeros([2**len(G.nodes), 2**len(G.nodes)])
    for i in range(0, len(G.nodes)):
        init[i] = Z    
        total += G.degree[len(G.nodes) - 1 - i]*kron(init)
        # reset
        init = [I]*len(G.nodes)
    
    total = np.array(total).diagonal()
    eigdz = np.diag(np.exp(np.complex(0,1)*gamma*total))
    rho = np.asmatrix(np.conj(eigdz))*rho*np.asmatrix(eigdz)

    init = [I]*len(G.nodes)
    total = complex(1,0)*np.zeros([2**len(G.nodes), 2**len(G.nodes)])
    for edge in G.edges:
        init[len(G.nodes) - 1 - edge[0]] = Z
        init[len(G.nodes) - 1 - edge[1]] = Z
        total += kron(init)
        # reset
        init = [I]*len(G.nodes)
        
    total = np.array(total).diagonal()
    eigzz = np.diag(np.exp(np.complex(0,1)*gamma*total))
    rho = np.asmatrix(np.conj(eigzz))*rho*np.asmatrix(eigzz)
    return rho

def ring_mixer(rho, G, beta):
    init = [I]*len(G.nodes)
    total = complex(1,0)*np.zeros([2**len(G.nodes), 2**len(G.nodes)])
    for i in range(0, len(G.nodes)):
        # X_i X_{i+1}
        init[i] = X 
        init[(i+1) % len(G.nodes)] = X
        total += kron(init)
        init = [I]*len(G.nodes)
        # Y_i Y_{i+1}
        init[i] = Y 
        init[(i+1) % len(G.nodes)] = Y
        total += kron(init)
        init = [I]*len(G.nodes)
    
    eibxxyy = expm(np.complex(0,1)*beta*total)
    rho = np.asmatrix(np.conj(eibxxyy))*np.asmatrix(rho)*np.asmatrix(eibxxyy)
    return rho

def prep(rho, G, k):
    for i in range(0, 2**len(G.nodes)):
        if num_ones(i) == k:
            rho[i, i] = 1
    #rho = rho/int(comb(len(G.nodes), k))
    return rho

def qaoa(G, k, gamma, beta0, beta1, p):
    rho = np.asmatrix(np.zeros([2**len(G.nodes), 2**len(G.nodes)]))
    rho = prep(rho, G, k)
    rho = ring_mixer(rho, G, beta0)
    for i in range(p):
        rho = phase_separator(rho, G, gamma)
        rho = ring_mixer(rho, G, beta1)
    return rho

def opt(args, *Gkp):
    rho = qaoa(Gkp[0], Gkp[1], args[0], args[1], args[2], Gkp[2])
    exp = expectation(Gkp[0], rho)
    return -exp

def ring_ps_ring(G, k, p, num_steps):
    # density matrix version
    gamma = pi/(2*(num_steps+1))
    beta0 = pi/(num_steps+1)
    beta1 = pi/(num_steps+1)
    exp_c = 0
    exp_arr = []
    angle_arr = []
    angles = []

    for i in range(0, num_steps):
        for j in range(0, num_steps):
            for l in range(0, num_steps):
                optimal = minimize(opt, [gamma, beta0, beta1], args=(G, k, p), bounds=[[0,pi/2],[0,pi],[0,pi]])
                if exp_c > optimal.fun:
                    exp_c = optimal.fun
                    angles = []
                    angles.append(gamma)
                    angles.append(beta0)
                    angles.append(beta1)
                print(-1*exp_c)
                gamma += pi/(2*(num_steps+1))
            gamma = pi/(2*(num_steps+1))
            beta0 += pi/(num_steps+1)
        beta0 = pi/(num_steps+1)
        beta1 += pi/(num_steps+1)

    return -1*exp_c, angle_arr

'''
def ring_ps_ring_old(G, k, p, num_steps):
    gamma = pi/(2*(num_steps+1))
    beta0 = pi/(num_steps+1)
    beta1 = pi/(num_steps+1)
    exp_c = 0
    exp_arr = []
    angle_arr = []
    angles = []

    num_init = int(comb(len(G.nodes), k))
    for m in range(0, num_init):
        for i in range(0, num_steps):
            for j in range(0, num_steps):
                for l in range(0, num_steps):
                    optimal = minimize(opt, [gamma, beta0, beta1], args=(G, k, p, m), bounds=[[0,pi/2],[0,pi],[0,pi]])
                    if exp_c > optimal.fun:
                        exp_c = optimal.fun
                        angles = []
                        angles.append(gamma)
                        angles.append(beta0)
                        angles.append(beta1)
                    gamma += pi/(2*(num_steps+1))
                gamma = pi/(2*(num_steps+1))
                beta0 += pi/(num_steps+1)
            beta0 = pi/(num_steps+1)
            beta1 += pi/(num_steps+1)
        exp_arr.append(exp_c*-1)
        exp_c = 0
        angle_arr.append(angles)

    avg = 0
    for entry in exp_arr:
        avg += entry/num_init
    return avg, angle_arr
'''

if __name__ == '__main__':
    G = nx.graph_atlas(6)
    ring_ps_ring(G, int(len(G.nodes)/2), 1, 3)

    rho = qaoa(Gkp[0], Gkp[1], args[0], args[1], args[2], Gkp[2])
    exp = expectation(Gkp[0], rho)
