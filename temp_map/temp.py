import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import pi
import sys
sys.path.insert(0, '../common/')
import common
import datetime
import random
import os
from multiprocessing import Process

G = None

def qaoa(gamma, beta, G, C, M):
    state = common.dicke(len(G.nodes))
    state = common.phase_separator(state, C, gamma)
    state = common.mixer(state, M, beta)
    return common.expectation(G, state)

def draw():
    global G
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def temp_map():
    global G
    for gi in range(91, 92):
        print('\t' + str(gi))
        #G = nx.gnp_random_graph(7, 0.5)
        G, C, M, k = common.get_complete(gi)

        #p = Process(target=draw)
        #p.start()

        num_steps = 30
        gamma, beta = 0, 0
        g_max = 2*pi
        b_max = pi/2
        g_list, grid = [], []
        grid_max = 0
        fig, ax = plt.subplots()

        print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
        for i in range(0, num_steps):
            for j in range(0, num_steps):
                val = qaoa(gamma, beta, G, C, M)
                g_list.append(val)
                if grid_max < val: grid_max = val
                gamma += g_max/(num_steps-1)
            beta += b_max/(num_steps-1)
            gamma = 0
            grid.append(g_list)
            g_list = []
            print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))

        grid = list(reversed(grid))
        print('-------------- max grid <C>: ' + str(grid_max))

        SIZE = 11
        plt.rc('font', size=SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SIZE)    # legend fontsize
        plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title


        im = ax.imshow(grid, aspect='auto', extent=(0, 2*pi, 0, pi/2), interpolation='gaussian', cmap=cm.inferno_r)
        cbar = ax.figure.colorbar(im, ax=ax, ticks=[])
        #cbar.ax.set_ylabel('$\\langle H_P \\rangle$', rotation=0, va="bottom")
        cbar.ax.get_yaxis().labelpad = 25
        cbar.ax.set_ylabel('$\\langle H_P \\rangle$', rotation=0)

        plt.xlabel('$\\gamma\,/\,\pi$')
        plt.ylabel('$\\beta\,/\,\pi$')
        #plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(len(G.nodes)) + ', k=' + str(k) + \
        #', p=' + str(1) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps) + ', gi=' + str(gi))
        plt.show()
        #plt.savefig('all/' + str(gi) + '.svg')
        #if flag == 0: plt.savefig('figures/p/' + str(gi) + '.png')
        #else: plt.savefig('figures/e/' + str(gi) + '.png')

if __name__ == '__main__':
    temp_map()
