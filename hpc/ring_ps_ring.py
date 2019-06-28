# Importing standard Qiskit libraries and configuring account
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.special import comb
from scipy.optimize import minimize
from math import pi
import datetime
from scipy.linalg import expm
from multiprocessing import Process
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

def dicke(state, G, k):
    for i in range(0, 2**len(G.nodes)):
        if num_ones(i) == k:
            state[i] = 1/(np.sqrt(comb(len(G.nodes), k)))
    return state

def expectation(G, state):
    total = 0
    for i in range(0, len(state)):
        if state[i] == 0:
            continue
        sa = [0]*len(G.nodes)
        b = bin(i)[2:]
        for j in range(0, len(b)):
            sa[j] = int(b[len(b)-1-j])
        total_cost = 0
        for edge in G.edges:
            cost = 3
            if sa[edge[0]] == 1 or sa[edge[1]] == 1:
                cost = -1
            total_cost += cost
        f = -(1/4)*(total_cost - 3*G.number_of_edges())
        prob = np.real(state[i]*np.conj(state[i]))
        total += f*prob
    return total

def phase_separator(state, G, gamma):
    init = [I]*len(G.nodes)
    # simplify later to diagonal matrix
    total = [complex(0,0)*np.matrix('0 0; 0 0')]*len(G.nodes)
    total = kron(total)
    for i in range(0, len(G.nodes)):
        init[i] = Z    
        total += G.degree[len(G.nodes) - 1 - i]*kron(init)
        # reset
        init = [I]*len(G.nodes)
    
    total = np.array(total).diagonal()
    eigdz = np.exp(np.complex(0,-1)*gamma*total)
    state = np.multiply(eigdz, state)

    init = [I]*len(G.nodes)
    total = [complex(0,0)*np.matrix('0 0; 0 0')]*len(G.nodes)
    total = kron(total)
    for edge in G.edges:
        init[len(G.nodes) - 1 - edge[0]] = Z
        init[len(G.nodes) - 1 - edge[1]] = Z
        total += kron(init)
        # reset
        init = [I]*len(G.nodes)
        
    total = np.array(total).diagonal()
    eigzz = np.exp(np.complex(0,-1)*gamma*total)
    state = np.multiply(eigzz, state)
    return state

def ring_mixer(state, G, beta):
    init = [I]*len(G.nodes)
    total = [complex(1,0)*np.matrix('0 0; 0 0')]*len(G.nodes)
    total = kron(total)
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
    
    eibxxyy = np.asmatrix(expm(np.complex(0,-1)*beta*total))
    state = np.dot(eibxxyy, state)
    state = np.asarray(state).reshape(-1)
    return state

def prep(state, G, k, m):
    counter = 0
    for i in range(0, 2**len(G.nodes)):
        if num_ones(i) == k:
            # start with different initial states
            if m == counter:
                state[i] = 1
                break
            counter += 1
    return state

def qaoa(G, k, gamma, beta0, beta1, p, m):
    state = np.zeros(2**len(G.nodes))
    state = prep(state, G, k, m)
    state = ring_mixer(state, G, beta0)
    for i in range(p):
        state = phase_separator(state, G, gamma)
        state = ring_mixer(state, G, beta1)

    # set small components to 0 for readability
    tol = 1e-16
    state.real[abs(state.real) < tol] = 0.0
    state.imag[abs(state.imag) < tol] = 0.0
    return state

def opt(args, *Gkpm):
    state = qaoa(Gkpm[0], Gkpm[1], args[0], args[1], args[2], Gkpm[2], Gkpm[3])
    exp = expectation(Gkpm[0], state)
    return -exp

def ring_ps_ring(G, k, p, num_steps):
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
