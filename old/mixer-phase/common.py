import numpy as np
import networkx as nx
from scipy.special import comb
from scipy.linalg import expm
import random
from itertools import combinations

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

def prob(G, state, gi):
    total = 0
    best = brute_force(gi)
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
        if f == best:
            total += prob
    return total

def create_C(G):
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
    return C.diagonal()

def create_ring_M(n):
    init = [I]*n
    M = complex(1,0)*np.zeros([2**n, 2**n])
    for i in range(0, n):
        # X_i X_{i+1}
        init[i] = X
        init[(i+1) % n] = X
        M += kron(init)
        init = [I]*n
        # Y_i Y_{i+1}
        init[i] = Y
        init[(i+1) % n] = Y
        M += kron(init)
        init = [I]*n
        if n == 2: break
    return M

def create_complete_M(n):
    init = [I]*n
    M = complex(1,0)*np.zeros([2**n, 2**n])
    for i in range(0, n):
        for j in range(0, n):
            if i > j:
                # X_i X_{j}
                init[i] = X
                init[j] = X
                M += kron(init)
                init = [I]*n
                # Y_i Y_{j}
                init[i] = Y
                init[j] = Y
                M += kron(init)
                init = [I]*n
    return M

def phase_separator(state, C, gamma):
    eiC = np.exp(np.complex(0,-1)*gamma*C)
    return np.multiply(eiC, state)

def mixer(state, M, beta):
    eibxxyy = expm(np.complex(0,-1)*beta*M)
    return np.matmul(eibxxyy, state)

def get_stuff(gi):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    C = create_C(G)
    M = create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)
    return G, C, M, k

def get_stuff_ring(gi):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    C = create_C(G)
    M = create_ring_M(len(G.nodes))
    k = int(len(G.nodes)/2)
    return G, C, M, k

def prep(state, G, k, m):
    counter = 0
    for i in range(0, 2**n):
        if num_ones(i) == k:
            # start with different initial-vs-dicke states
            if m == counter:
                state[i] = 1
                break
            counter += 1
    return state

def dicke(n, k):
    state = np.zeros(2**n)
    for i in range(0, 2**n):
        if num_ones(i) == k:
            state[i] = 1/(np.sqrt(comb(n, k)))
    return state

def MLHS(p, num_samples, lower_g, upper_g, lower_b, upper_b):
    range_g = upper_g - lower_g
    range_b = upper_b - lower_b
    g = [(lower_g + (range_g/num_samples)*(i + 0.5)) for i in range(num_samples)]
    b = [(lower_b + (range_b/num_samples)*(i + 0.5)) for i in range(num_samples)]
    valid_g = [[i for i in range(num_samples)] for j in range(p)]
    valid_b = [[i for i in range(num_samples)] for j in range(p)]
    sample_g = [[] for j in range(num_samples)]
    sample_b = [[] for j in range(num_samples)]
    for j in range(p):
        for i in range(num_samples):
            vg = random.randint(0, len(valid_g[j])-1)
            vb = random.randint(0, len(valid_b[j])-1)
            sample_g[i].append(g[valid_g[j][vg]])
            sample_b[i].append(b[valid_b[j][vb]])
            del valid_g[j][vg]
            del valid_b[j][vb]
    return sample_g, sample_b

def brute_force(gi):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
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

def random_k_state(n, k):
    state = np.zeros(2**n)
    num_k = int(comb(n, k))
    index = random.randint(0, num_k-1)
    counter = 0
    for i in range(0, 2**n):
        if num_ones(i) == k:
            # start with different initial-vs-dicke states
            if index == counter:
                state[i] = 1
                break
            counter += 1
    return state

# pick an m in range: [0, (n choose k)-1]
def select_k_state(n, k, m):
    state = np.zeros(2**n)
    counter = 0
    for i in range(0, 2**n):
        if num_ones(i) == k:
            # start with different initial-vs-dicke states
            if m == counter:
                state[i] = 1
                break
            counter += 1
    return state
