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
        f = -(0.25)*(total_cost - 3*len(G.edges))
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

def qaoa(G, k, gamma, beta, p):
    state = np.zeros(2**len(G.nodes))
    state = dicke(state, G, k)
    for i in range(p):
        state = phase_separator(state, G, gamma)
        state = ring_mixer(state, G, beta)

    # set small components to 0 for readability
    tol = 1e-16
    state.real[abs(state.real) < tol] = 0.0
    state.imag[abs(state.imag) < tol] = 0.0
    return state

def opt(args, *Gkp):
    state = qaoa(Gkp[0], Gkp[1], args[0], args[1], Gkp[2])
    exp = expectation(Gkp[0], state)
    return -exp

def dicke_ps_ring(G, k, p, num_steps):
    gamma = pi/(2*(num_steps+1))
    beta = pi/(num_steps+1)
    exp_c = 0
    angles = []

    for i in range(0, num_steps):
        for j in range(0, num_steps):
            optimal = minimize(opt, [gamma, beta], args=(G, k, p,), bounds=[[0,pi/2],[0,pi]])
            if exp_c > optimal.fun:
                exp_c = optimal.fun
                angles = []
                angles.append(gamma)
                angles.append(beta)
            gamma += pi/(2*(num_steps+1))
        beta += pi/(num_steps+1)
        gamma = pi/(2*(num_steps+1))
    return exp_c*-1, angles
