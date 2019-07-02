import networkx as nx
import numpy as np
from scipy.misc import comb
from scipy.optimize import minimize
from math import pi
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Global variables
I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1],[1,0]])*complex(0,1)
Z = np.array([[1,0],[0,-1]])

# take the kronecker (tensor) product of a list of matrices
def kron(m):
    if len(m) == 1:
        return m
    total = np.kron(m[1], m[0])
    if len(m) > 2:
        for i in range(2, len(m)):
            total = np.kron(m[i], total)
    return total

def num_ones(n):
    c = 0
    while n:
        c += 1
        n &= n - 1
    return c

def expectation(G, rho):
    init = [I]*len(G.nodes)
    C = complex(1,0)*np.zeros([2**len(G.nodes), 2**len(G.nodes)])
    for i in range(0, len(G.nodes)):
        init[i] = Z
        C += G.degree[i]*kron(init)
        # reset
        init = [I]*len(G.nodes)

    init = [I]*len(G.nodes)
    for edge in G.edges:
        init[edge[0]] = Z
        init[edge[1]] = Z
        C += kron(init)
        # reset
        init = [I]*len(G.nodes)

    #C_exp = np.matmul(np.transpose(np.conj(state)), np.matmul(C, state))
    C_exp = np.trace(np.matmul(C, rho))
    return np.real(-(0.25)*(C_exp - 3*G.number_of_edges()))

def phase_separator(rho, G, gamma):
    init = [I]*len(G.nodes)
    total = complex(1,0)*np.zeros([2**len(G.nodes), 2**len(G.nodes)])
    for i in range(0, len(G.nodes)):
        init[i] = Z
        total += G.degree[i]*kron(init)
        init = [I]*len(G.nodes)
    
    eigdz = np.diag(np.exp(np.complex(0,-1)*gamma*total.diagonal()))
    rho = np.matmul(eigdz, np.matmul(rho, np.transpose(np.conj(eigdz))))
    rho.real[abs(rho.real) < 1e-16] = 0.0
    rho.imag[abs(rho.imag) < 1e-16] = 0.0

    init = [I]*len(G.nodes)
    total = complex(1,0)*np.zeros([2**len(G.nodes), 2**len(G.nodes)])
    for edge in G.edges:
        init[edge[0]] = Z
        init[edge[1]] = Z
        total += kron(init)
        init = [I]*len(G.nodes)
        
    eigzz = np.diag(np.exp(np.complex(0,-1)*gamma*total.diagonal()))
    rho = np.matmul(eigzz, np.matmul(rho, np.transpose(np.conj(eigzz))))
    rho.real[abs(rho.real) < 1e-16] = 0.0
    rho.imag[abs(rho.imag) < 1e-16] = 0.0
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
    
    eibxxyy = expm(np.complex(0,-1)*beta*total)
    rho = np.matmul(eibxxyy, np.matmul(rho, np.transpose(np.conj(eibxxyy))))
    rho.real[abs(rho.real) < 1e-16] = 0.0
    rho.imag[abs(rho.imag) < 1e-16] = 0.0
    return rho

def prep(rho, G, k):
    for i in range(0, 2**len(G.nodes)):
        if num_ones(i) == k:
            rho[i, i] = 1
    rho = rho/int(comb(len(G.nodes), k))
    return rho

def qaoa(G, k, gamma, beta0, beta1, p):
    rho = np.zeros([2**len(G.nodes), 2**len(G.nodes)])
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
    angle_arr = []

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
                gamma += pi/(2*(num_steps+1))
            gamma = pi/(2*(num_steps+1))
            beta0 += pi/(num_steps+1)
        beta0 = pi/(num_steps+1)
        beta1 += pi/(num_steps+1)

    return -1*exp_c, angle_arr

if __name__ == '__main__':
    G = nx.graph_atlas(6)
    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.show()
    exp, angles = ring_ps_ring(G, int(len(G.nodes)/2), 1, 3)
    print('exp: ' + str(exp))
