# Importing standard Qiskit libraries and configuring account
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import pi
import datetime

# Global variables
I = np.matrix('1 0; 0 1')
X = np.matrix('0 1; 1 0')
Y = np.matrix('0 -1; 1 0')*complex(0,1)
Z = np.matrix('1 0; 0 -1')
paulis = [I,X,Y,Z]


num_nodes = 4
num_shots = 1000
k = 1

def generate_graph():
    # Generate a random graph
    #G = nx.fast_gnp_random_graph(num_nodes,0.8)
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0,1),(1, 2), (1, 3), (0,3)])
    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.show()
    return G

# take the kronecker (tensor) product of a list of len(m) matrices
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
            state[i] = 1
    return state

def expectation(G, counts):
    total = 0
    for state in counts:
        sa = []
        for i in state:
            sa.append(int(i))
        sa = list(reversed(sa))
        total_cost = 0
        for edge in G.edges:
            cost = 3
            if sa[edge[0]] == 1 or sa[edge[1]] == 1:
                cost = -1
            total_cost += cost
        f = -(1/4)*(total_cost - 3*G.number_of_edges())
        total += f*(counts[state]/num_shots)
    return total

def phase_separator(state, G, gamma):
    # e^{-i \gamma deg(j) Z_j}}
    init = [I]*len(G.nodes)
    # simplify later to diagonal matrix
    total = [np.matrix('0 0; 0 0')]*len(G.nodes)
    total = kron(total)
    for i in range(0, len(G.nodes)):
        init[i] = Z    
        total += G.degree[i]*kron(init)
        # reset
        init = [I]*len(G.nodes)
    
    total = np.array(total).diagonal()
    eigdz = np.exp(np.complex(0,-1)*gamma*total)
    state = np.multiply(eigdz, state)

    init = [I]*len(G.nodes)
    total = [np.matrix('0 0; 0 0')]*len(G.nodes)
    total = kron(total)
    for edge in G.edges:
        init[edge[0]] = Z
        init[edge[1]] = Z
        total += kron(init)
        # reset
        init = [I]*len(G.nodes)
        
    total = np.array(total).diagonal()
    eigzz = np.exp(np.complex(0,-1)*gamma*total)
    state = np.multiply(eigzz, state)
    return state

def ring_mixer(G, circ, beta):
    # even terms
    for i in range(0, len(G.nodes)-1, 2):
        #print(str(i) + ", " + str((i+1) % len(G.nodes)))
        eix(circ, beta, i, (i+1) % len(G.nodes))
        eiy(circ, beta, i, (i+1) % len(G.nodes))

    # odd terms
    for i in range(1, len(G.nodes), 2):
        #print(str(i) + ", " + str((i+1) % len(G.nodes)))
        eix(circ, beta, i, (i+1) % len(G.nodes))
        eiy(circ, beta, i, (i+1) % len(G.nodes))

    # if number of edges in ring is odd, we have one leftover term
    if len(G.nodes) % 2 != 0:
        #print("special case")
        #print(str(len(G.nodes)-1) + ", 0")
        eix(circ, beta, len(G.nodes)-1, 0)
        eiy(circ, beta, len(G.nodes)-1, 0)

def qaoa(G, gamma, beta, p):
    state = np.zeros(2**len(G.nodes))
    # prepare equal superposition over Hamming weight k
    state = dicke(state, G, k)
    state = phase_separator(state, G, 0.5)
    print(state)
    '''
    for i in range(p):
        state = phase_separator(state, G, gamma)
        state = ring_mixer(state, G, beta)
    '''
    return 0

def gamma_beta():
    G = generate_graph()
    num_steps = 100
    gamma = 0
    beta = 0
    p = 1
    g_list = []
    grid = []
    fig, ax = plt.subplots()

    print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
    for i in range(0, num_steps):
        for j in range(0, num_steps):
            counts = qaoa(G, gamma, beta, p)
            exp = expectation(G, counts)
            g_list.append(exp)
            #print('g: ' + str(gamma) + ', b: ' + str(beta) + ', exp: ' + str(exp))
            gamma += pi/(num_steps-1)
        beta += pi/(num_steps-1)
        gamma = 0
        grid.append(g_list)
        g_list = []
        print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))

    grid = list(reversed(grid))
    #print(grid)

    im = ax.imshow(grid, extent=(0, pi, 0, pi), interpolation='gaussian', cmap=cm.inferno_r)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('$\\langle C \\rangle$', rotation=-90, va="bottom")

    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\beta$')
    plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(num_nodes) + ', k=' + str(k) + ', p=' + str(p) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps))
    plt.show()

if __name__ == '__main__':
    gamma_beta()
