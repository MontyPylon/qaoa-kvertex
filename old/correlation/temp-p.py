import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import pi
import datetime
import os
import random

import sys
sys.path.insert(0, '/home/montypylon/lanl/qaoa-kvertex/mixer-phase')
import dicke_ps_complete, common

def temp_map(G, k, p, gi, exp_opt, prev, angles):
    num_steps = 30
    gamma = 0
    beta = 0
    g_list, grid, graph_angles = [], [], []
    grid_max = 0
    fig, ax = plt.subplots()
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))

    print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
    for i in range(0, num_steps):
        for j in range(0, num_steps):
            if p > 1: graph_angles.extend(prev[0])
            graph_angles.append(gamma)
            if p > 1: graph_angles.extend(prev[1])
            graph_angles.append(beta)
            state = dicke_ps_complete.qaoa([G, C, M, k, p], graph_angles)
            exp = common.expectation(G, state)
            g_list.append(exp)
            if grid_max < exp:
                grid_max = exp
            gamma += pi/(2*(num_steps-1))
            graph_angles = []
        beta += pi/(num_steps-1)
        gamma = 0
        grid.append(g_list)
        g_list = []
        print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))

    grid = list(reversed(grid))
    print('-------------- max grid <C>: ' + str(grid_max))
    print('-------------- max optm <C>: ' + str(exp_opt))

    im = ax.imshow(grid, aspect='auto', extent=(0, pi/2, 0, pi), interpolation='gaussian', cmap=cm.inferno_r)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('$\\langle C \\rangle$', rotation=-90, va="bottom")

    axes = plt.gca()
    axes.set_ylim([0, pi])
    axes.set_xlim([0, pi/2])

    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\beta$')
    plt.title('$\\beta \\ vs \\ \\gamma$, opt_exp=' + str(exp_opt) + '\nn=' + str(len(G.nodes)) + ', k=' + str(k) + \
              ', p=' + str(p) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps) + ', gi=' + str(gi))

    plt.scatter(angles[0], angles[1], s=50, c='yellow', marker='o')
    #plt.show()
    path = '3-reg/' + str(gi) + '/'
    if not os.path.exists(path): os.mkdir(path)
    plt.savefig(path + str(p) + '.png')
    plt.cla()

if __name__ == '__main__':
    #for gi in range(600, 640):
    #G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
    #gi = random.randint(1001, 10000)
    #G = nx.fast_gnp_random_graph(10, 0.5)
    #while not nx.is_connected(G):
    #    G = nx.fast_gnp_random_graph(10, 0.5)

    G = nx.random_regular_graph(3, 8)
    gi = random.randint(1,1000)
    nx.draw(G, with_labels=True, font_weight='bold')
    path = '3-reg/' + str(gi) + '/'
    if not os.path.exists(path): os.mkdir(path)
    plt.savefig(path + 'graph.png')
    plt.cla()

    print('starting')
    k = int(len(G.nodes)/2)
    num_samples = 10

    max_p = 5
    prev_g = []
    prev_b = []
    for p in range(1, max_p+1):
        exp, angles = dicke_ps_complete.restricted_MLHS(G, k, p, num_samples, [prev_g, prev_b])
        print('exp = ' + str(exp))
        print('angles = ' + str(angles))
        temp_map(G, k, p, gi, exp, [prev_g, prev_b], angles)
        prev_g.append(angles[0])
        prev_b.append(angles[1])
