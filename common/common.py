import numpy as np
import networkx as nx
from scipy.special import comb
from scipy.linalg import expm
import random
from itertools import combinations

def expectation(G, k, state):
    total = 0
    values = []
    for i in range(2**(len(G.nodes))):
        if num_ones(i) == k: values.append(i)

    for i in range(len(values)):
        sa = [0]*len(G.nodes)
        b = bin(values[i])[2:]
        for j in range(0, len(b)):
            sa[j] = int(b[len(b)-1-j])
        f = 0
        for edge in G.edges:
            if sa[edge[0]] == 1 or sa[edge[1]] == 1: f += 1
        prob = np.real(state[i]*np.conj(state[i]))
        total += f*prob
    return total

def prob(G, k, state):
    k = int(len(G.nodes)/2)
    total = 0
    best = brute_force(G)
    values = []
    for i in range(2**(len(G.nodes))):
        if num_ones(i) == k: values.append(i)

    for i in range(len(values)):
        sa = [0]*len(G.nodes)
        b = bin(values[i])[2:]
        for j in range(0, len(b)):
            sa[j] = int(b[len(b)-1-j])
        f = 0
        for edge in G.edges:
            if sa[edge[0]] == 1 or sa[edge[1]] == 1: f += 1
        prob = np.real(state[i]*np.conj(state[i]))
        if f == best:
            total += prob
    return total

def create_C(G, k):
    n = len(G.nodes)
    size = int(comb(n, k))
    diag = np.zeros((size))

    values = []
    for i in range(2**n):
        if num_ones(i) == k: values.append(i)

    for i in range(len(values)):
        sa = [0]*len(G.nodes)
        b = bin(values[i])[2:]
        for j in range(0, len(b)):
            sa[j] = int(b[len(b)-1-j])
        f = 0
        for edge in G.edges:
            if sa[edge[0]] == 1 or sa[edge[1]] == 1: f += 1
        diag[i] = f

    return diag

def num_ones(n):
    c = 0
    while n:
        c += 1
        n &= n - 1
    return c

def next_to(n, i):
    b = bin(i)[2:]
    while len(b) != n:
        b = '0' + b
    for ii in range(len(b)):
        if b[ii] == '1' and b[(ii+1) % len(b)] == '1': return True
    return False

def create_ring_M(n, k):
    size = int(comb(n, k))
    m = complex(1,0)*np.zeros((size, size))
    values = []
    for i in range(2**n):
        if num_ones(i) == k: values.append(i)

    for i in range(len(values)):
        for j in range(len(values)):
            if j > i:
                if num_ones(values[i]^values[j]) == 2:
                    if next_to(n, values[i] ^ values[j]):
                        m[i,j] = 2
                        m[j,i] = 2
    return m

def create_complete_M(n, k):
    size = int(comb(n, k))
    m = complex(1,0)*np.zeros((size, size))
    values = []

    for i in range(2**n):
        if num_ones(i) == k: values.append(i)

    for i in range(len(values)):
        for j in range(len(values)):
            if j > i:
                if num_ones(values[i]^values[j]) == 2:
                    m[i,j] = 2
                    m[j,i] = 2
    return m

def phase_separator(state, C, gamma):
    eiC = np.exp(np.complex(0,-1)*gamma*C)
    return np.multiply(eiC, state)

def mixer(state, M, beta):
    eibxxyy = expm(np.complex(0,-1)*beta*M)
    return np.matmul(eibxxyy, state)

def get_complete(gi):
    G = nx.read_gpickle('../graphs/atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
    C = create_C(G, k)
    M = create_complete_M(len(G.nodes), k)
    return G, C, M, k

def get_ring(gi):
    G = nx.read_gpickle('../graphs/atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
    C = create_C(G, k)
    M = create_ring_M(len(G.nodes), k)
    return G, C, M, k

def dicke(n, k):
    size = int(comb(n, k))
    state = np.ones(size)*(1/np.sqrt(size))
    return state

def random_k_state(n, k):
    size = int(comb(n, k))
    state = np.zeros(size)
    index = random.randint(0, size-1)
    state[index] = 1
    return state

def MLHS(p, num_samples, lower_g, upper_g, lower_b, upper_b):
    range_g = upper_g - lower_g
    range_b = upper_b - lower_b
    g = [(lower_g + (range_g/num_samples)*(i + 0.5)) for i in range(num_samples)]
    b = [(lower_b + (range_b/num_samples)*(i + 0.5)) for i in range(num_samples)]
    valid_g = [[i for i in range(num_samples)] for _ in range(p)]
    valid_b = [[i for i in range(num_samples)] for _ in range(p)]
    sample_g = [[] for _ in range(num_samples)]
    sample_b = [[] for _ in range(num_samples)]
    for j in range(p):
        for i in range(num_samples):
            vg = random.randint(0, len(valid_g[j])-1)
            vb = random.randint(0, len(valid_b[j])-1)
            sample_g[i].append(g[valid_g[j][vg]])
            sample_b[i].append(b[valid_b[j][vb]])
            del valid_g[j][vg]
            del valid_b[j][vb]
    return sample_g, sample_b

def brute_force(G, k):
    comb = combinations(G.nodes, k)
    highest = 0
    for group in list(comb):
        score = 0
        for edge in G.edges:
            for v in group:
                if v == edge[0] or v == edge[1]:
                    score += 1
                    break
        if score > highest:
            highest = score
    return highest

'''
if __name__ == '__main__':
    gi = random.randint(143, 955)
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    create_ring_M(4)
    #create_complete_M(3, 1)
'''
