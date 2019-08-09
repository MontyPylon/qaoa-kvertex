import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import pi
import sys
sys.path.insert(0, '../faster/')
import faster
import datetime
import random
import os
from multiprocessing import Process

G = None

def qaoa(gamma, beta, G, C, M, k):
    state = faster.dicke(len(G.nodes), k)
    state = faster.phase_separator(state, C, gamma)
    state = faster.mixer(state, M, beta)
    return faster.expectation(G, k, state)

def draw():
    global G
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def temp_map():
    global G
    total = []
    #G = nx.gnp_random_graph(7, 0.6)
    for ii in range(1,2):
        gi = random.randint(163, 955)
        print(gi)
        G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
        k = int(len(G.nodes)/2)
        C = faster.create_C(G, k)
        M = faster.create_complete_M(len(G.nodes), k)
        #M = common.create_ring_M(len(G.nodes))

        p = Process(target=draw)
        p.start()


        num_steps = 50
        gamma, beta = 0, 0
        g_list, grid = [], []
        grid_min = -1
        grid_max = 0
        fig, ax = plt.subplots()

        print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
        for i in range(0, num_steps):
            for j in range(0, num_steps):
                val = qaoa(gamma, beta, G, C, M, k)
                g_list.append(val)
                if grid_max < val: grid_max = val
                if grid_min == -1: grid_min = val
                if grid_min > val: grid_min = val
                gamma += 2*pi/(num_steps-1)
            beta += pi/(2*(num_steps-1))
            gamma = 0
            grid.append(g_list)
            g_list = []
            print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))

        grid = list(reversed(grid))
        print('-------------- max grid <C>: ' + str(grid_max))
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                grid[i][j] = (1/(grid_max-grid_min))*(grid[i][j] - grid_min)

        if not total:
            total = grid
        else:
            for i in range(len(total)):
                for j in range(len(total[0])):
                    total[i][j] += grid[i][j]

        temp = total
        '''
        temp_max = -1
        temp_min = -1
        for i in range(len(total)):
            for j in range(len(total[0])):
                if temp_max == -1: temp_max = temp[i][j]
                if temp[i][j] > temp_max: temp_max = temp[i][j]
                if temp_min == -1: temp_min = temp[i][j]
                if temp[i][j] < temp_min: temp_min = temp[i][j]
                #temp[i][j] = (1/ii)*temp[i][j]
        for i in range(len(total)):
            for j in range(len(total[0])):
                temp[i][j] = (1/(temp_max-temp_min))*(temp[i][j] - temp_min)
        '''

        SIZE = 13
        plt.rc('font', size=SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SIZE)    # legend fontsize
        plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title


        im = ax.imshow(temp, aspect='auto', extent=(0, 2, 0, 0.5), interpolation='gaussian', cmap=cm.inferno_r)
        cbar = ax.figure.colorbar(im, ax=ax, ticks=[])
        #cbar.ax.set_ylabel('$\\langle H_P \\rangle$', rotation=0, va="bottom")
        cbar.ax.get_yaxis().labelpad = 28
        cbar.ax.set_ylabel('$\\langle H_P \\rangle$', rotation=0, fontsize=13)

        plt.xlabel('$\\gamma\,/\,\pi$')
        plt.ylabel('$\\beta\,/\,\pi$')
        plt.tight_layout()
        #plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(len(G.nodes)) + ', k=' + str(k) + \
                  #', p=' + str(1) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps) + ', gi=' + str(gi))
        plt.show()
        #plt.savefig('overlay-fixed/' + str(ii) + '.svg')
        #if flag == 0: plt.savefig('figures/p/' + str(gi) + '.png')
        #else: plt.savefig('figures/e/' + str(gi) + '.png')

if __name__ == '__main__':
    temp_map()
