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
    state = common.dicke(state, G, k)
    for i in range(p):
        state = common.phase_separator(state, C, g_list[i])
        state = common.ring_mixer(state, M, b_list[i])
    return common.expectation(G, state)

if __name__ == '__main__':
    G = nx.read_gpickle('atlas/200.gpickle')
    C = common.create_C(G)
    M = common.create_M(G)
    k = int(len(G.nodes)/2)

    num_samples = 1000

    max_p = 20
    for p in range(1, max_p):
        found = False
        data, g_list, b_list = [], [], []
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
            for i in range(0, num_samples):
                for j in range(0, p):
                    g_list.append(random.uniform(0,pi/2))
                    b_list.append(random.uniform(0,pi))
                value = qaoa(p, g_list, b_list)
                data.append(value)
                g_list, b_list = [], []
                if i % 100 == 0:
                    print('Sample ' + str(i) + '\t' + str(datetime.datetime.now().time()))
                    pickle.dump(data, open('hist/' + str(p) + '.hist', 'wb'))

        avg = np.round(np.average(data), 3)
        std = np.round(np.std(data), 3)
        col = [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
        plt.hist(data, bins=100, range=[4,8], color=col)
        plt.axvline(x=np.max(data), color=col)
        plt.title('<C> histogram for p = ' + str(p) + ', gi=200, samples=' + str(num_samples) \
                  + '\navg=' + str(avg) + ', std=' + str(std))
        plt.xlabel('<C>')
        plt.ylabel('Counts of <C>')
        axes = plt.gca()
        axes.set_ylim([0, 130])
        #plt.savefig('hist/fig/p' + str(p) + '.png')
        plt.show()

