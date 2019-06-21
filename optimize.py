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

num_nodes = 5
G = None

def draw():
    global G
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def generate_graph():
    global G
    # Generate a random graph
    G = nx.fast_gnp_random_graph(num_nodes, random.random())
    while not nx.is_connected(G):
        G = nx.fast_gnp_random_graph(num_nodes, random.random())
    #G = nx.complete_graph(num_nodes)
    #G = nx.Graph()
    #G.add_nodes_from([0, 1, 2, 3])
    #G.add_edges_from([(0,1),(1, 2), (1, 3), (0,3)])
    p = Process(target=draw)
    p.start()
    return G

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

def eix(state, G, beta, i, j):
    init = [I]*len(G.nodes)
    init[len(G.nodes) - 1 - i] = X
    init[len(G.nodes) - 1 - j] = X
    total = kron(init)
    eibxx = np.asmatrix(expm(np.complex(0,-1)*beta*total))
    state = np.dot(eibxx, state)
    return np.asarray(state).reshape(-1)

def eiy(state, G, beta, i, j):
    init = [I]*len(G.nodes)
    init[len(G.nodes) - 1 - i] = Y
    init[len(G.nodes) - 1 - j] = Y
    total = kron(init)
    eibxx = np.asmatrix(expm(np.complex(0,-1)*beta*total))
    state = np.dot(eibxx, state)
    return np.asarray(state).reshape(-1)

def parity_ring_mixer(state, G, beta):
    # even terms
    for i in range(0, len(G.nodes)-1, 2):
        #print(str(i) + ", " + str((i+1) % len(G.nodes)))
        state = eix(state, G, beta, i, (i+1) % len(G.nodes))
        state = eiy(state, G, beta, i, (i+1) % len(G.nodes))

    # odd terms
    for i in range(1, len(G.nodes), 2):
        #print(str(i) + ", " + str((i+1) % len(G.nodes)))
        state = eix(state, G, beta, i, (i+1) % len(G.nodes))
        state = eiy(state, G, beta, i, (i+1) % len(G.nodes))

    # if number of edges in ring is odd, we have one leftover term
    if len(G.nodes) % 2 != 0:
        #print("special case")
        #print(str(len(G.nodes)-1) + ", 0")
        state = eix(state, G, beta, len(G.nodes)-1, 0)
        state = eiy(state, G, beta, len(G.nodes)-1, 0)

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
        #state = parity_ring_mixer(state, G, beta)

    # set small components to 0 for readability
    tol = 1e-16
    state.real[abs(state.real) < tol] = 0.0
    state.imag[abs(state.imag) < tol] = 0.0
    return state

def opt(args, *kp):
    global G
    state = qaoa(G, kp[0], args[0], args[1], kp[1])
    exp = expectation(G, state)
    return -exp


def gamma_beta():
    global G
    G = generate_graph()
    k = 1
    p = 1
    num_steps = 6
    gamma = pi/(2*(num_steps+1))
    beta = pi/(num_steps+1)
    opts = []
    exp_c = 0

    print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
    for i in range(0, num_steps):
        for j in range(0, num_steps):
            optimal = minimize(opt, [gamma, beta], args=(k, p,), bounds=[[0,pi/2],[0,pi]])
            opts.append(optimal.x)
            if exp_c > optimal.fun:
                exp_c = optimal.fun
            gamma += pi/(2*(num_steps+1))
        beta += pi/(num_steps+1)
        gamma = pi/(2*(num_steps+1))
        print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
    
    exp_c *= -1
    print('------------- max <C>: ' + str(exp_c))
    temp_map(k, p, opts, exp_c)

def temp_map(k, p, opts, exp_c):
    num_steps = 30
    gamma = 0
    beta = 0
    g_list = []
    grid = []
    grid_max = 0
    fig, ax = plt.subplots()

    print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
    for i in range(0, num_steps):
        for j in range(0, num_steps):
            state = qaoa(G, k, gamma, beta, p)
            exp = expectation(G, state)
            g_list.append(exp)
            if grid_max < exp:
                grid_max = exp
            #print('g: ' + str(gamma) + ', b: ' + str(beta) + ', exp: ' + str(exp))
            gamma += pi/(2*(num_steps-1))
        beta += pi/(num_steps-1)
        gamma = 0
        grid.append(g_list)
        g_list = []
        print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))

    grid = list(reversed(grid))
    print('-------------- max grid <C>: ' + str(grid_max))

    im = ax.imshow(grid, aspect='auto', extent=(0, pi/2, 0, pi), interpolation='gaussian', cmap=cm.inferno_r)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('$\\langle C \\rangle$', rotation=-90, va="bottom")

    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\beta$')
    plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(num_nodes) + ', k=' + str(k) + ', p=' + str(p) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps))

    for i in opts:
        plt.scatter(i[0], i[1], s=50, c='yellow', marker='o')
    plt.show()

if __name__ == '__main__':
    gamma_beta()
