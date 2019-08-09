import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import common
import pickle
import datetime
import random
import os

G, C, M, k = None, None, None, None


def probabilities(G, state):
    total = [0]*6
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
        #total += f*prob
        total[int(f-5)] += prob
    return total

def qaoa(p, g_list, b_list):
    global G, C, M, k
    state = np.zeros(2**len(G.nodes))
    state = common.dicke(state, len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, g_list[i])
        state = common.mixer(state, M, b_list[i])
    #return common.expectation(G, state)
    return probabilities(G, state), common.expectation(G, state)

def plot_hist(data, p, num_samples):
    avg = np.round(np.average(data), 3)
    std = np.round(np.std(data), 3)
    max = np.round(np.max(data), 3)
    col = [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
    plt.hist(data, bins=100, range=[5,10], color=col)
    plt.axvline(x=np.max(data), color=col)
    plt.axvline(x=avg, color=col)
    plt.title('<C> histogram for p = ' + str(p) + ', gi=502, samples=' + str(num_samples) \
              + '\navg=' + str(avg) + ', std=' + str(std) + ', max=' + str(max))
    plt.xlabel('<C>')
    plt.ylabel('Counts of <C>')
    axes = plt.gca()
    axes.set_ylim([0, 130])
    if not os.path.exists('hist/fig/'): os.mkdir('hist/fig/')
    #plt.savefig('hist/fig/p' + str(p) + '.png')
    plt.show()
    plt.cla()

def plot_val(data, p, num_samples):
    plt.bar([5,6,7,8,9,10], data)
    plt.title('<C> histogram for p = ' + str(p) + ', gi=502, samples=' + str(num_samples))
    plt.xlabel('Value of measured solution')
    plt.ylabel('Probability of measuring')
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    if not os.path.exists('hist/fig/'): os.mkdir('hist/fig/')
    plt.savefig('hist/fig/p' + str(p) + '.png')
    #plt.show()
    plt.cla()

if __name__ == '__main__':
    G = nx.read_gpickle('atlas/502.gpickle')
    C = common.create_C(G)
    M = common.create_ring_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    num_samples = 1000

    max_p = 10
    best_g, best_b = [], []
    for p in range(1, max_p+1):
        found = False
        data = [0]*6
        print('--------- p = ' + str(p) + '----------')
        path = 'hist/' + str(p) + '.hist'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                    found = True
                except Exception as e:
                    print('Error opening dictionary for ' + str(p) + '.hist')
                    print(e)
        if not found:
            g = [(pi/(2*num_samples))*(i + 0.5) for i in range(num_samples)]
            b = [(pi/num_samples)*(i + 0.5) for i in range(num_samples)]
            valid_g = [i for i in range(num_samples)]
            valid_b = [i for i in range(num_samples)]
            sample_g = [best_g.copy() for j in range(num_samples)]
            sample_b = [best_b.copy() for j in range(num_samples)]
            for i in range(num_samples):
                vg = random.randint(0, len(valid_g)-1)
                vb = random.randint(0, len(valid_b)-1)
                sample_g[i].append(g[valid_g[vg]])
                sample_b[i].append(b[valid_b[vb]])
                del valid_g[vg]
                del valid_b[vb]

            best_value = -1
            for i in range(0, num_samples):
                probs, value = qaoa(p, sample_g[i], sample_b[i])
                #data.append(value)
                for j in range(0,6):
                    data[j] += probs[j]
                if value > best_value:
                    best_value = value
                    best_g = sample_g[i]
                    best_b = sample_b[i]
                if i % 100 == 0:
                    print('Sample ' + str(i) + '\t' + str(datetime.datetime.now().time()))
                    if not os.path.exists('hist/'): os.mkdir('hist/')
                    pickle.dump(data, open('hist/' + str(p) + '.hist', 'wb'))

        # normalize to probability
        for j in range(len(data)): data[j] /= num_samples
        # plot
        plot_val(data, p, num_samples)


