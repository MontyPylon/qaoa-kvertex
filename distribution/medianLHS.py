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

def qaoa(p, g_list, b_list):
    global G, C, M, k
    state = np.zeros(2**len(G.nodes))
    state = common.dicke(state, len(G.nodes), k)
    for i in range(p):
        state = common.phase_separator(state, C, g_list[i])
        state = common.mixer(state, M, b_list[i])
    return common.expectation(G, state)

if __name__ == '__main__':
    G = nx.read_gpickle('atlas/502.gpickle')
    C = common.create_C(G)
    M = common.create_ring_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    num_samples = 1000

    max_p = 10
    for p in range(1, max_p+1):
        found = False
        data = []
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
            print('p = ' + str(p))
            g = [(pi/(2*num_samples))*(i + 0.5) for i in range(num_samples)]
            b = [(pi/num_samples)*(i + 0.5) for i in range(num_samples)]
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

            for i in range(0, num_samples):
                value = qaoa(p, sample_g[i], sample_b[i])
                data.append(value)
                if i % 100 == 0:
                    print('Sample ' + str(i) + '\t' + str(datetime.datetime.now().time()))
                    pickle.dump(data, open('hist/' + str(p) + '.hist', 'wb'))

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
        plt.savefig('hist/fig/p' + str(p) + '.png')
        #plt.show()
        plt.cla()

