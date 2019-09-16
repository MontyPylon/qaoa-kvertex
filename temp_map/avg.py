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
from matplotlib.ticker import FormatStrFormatter

G = None

def qaoa(gamma, beta, G, C, M, k):
    state = common.dicke(len(G.nodes), k)
    state = common.phase_separator(state, C, gamma)
    state = common.mixer(state, M, beta)
    return common.expectation(G, k, state)

def draw():
    global G
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def temp_map():
    global G
    total = []
    #G = nx.gnp_random_graph(7, 0.6)
    for ii in range(1,100):
        total = []
        #gi = random.randint(163, 955)
        #print(gi)
        #G = nx.read_gpickle('../graphs/atlas/' + str(gi) + '.gpickle')
        num_nodes = random.randint(5,10)
        G = nx.fast_gnp_random_graph(num_nodes, 0.5)
        while not nx.is_connected(G): G = nx.fast_gnp_random_graph(num_nodes, 0.5)
        k = int(len(G.nodes)/2)
        C = common.create_C(G, k)
        M = common.create_complete_M(len(G.nodes), k)
        #M = common.create_ring_M(len(G.nodes), k)

        #p = Process(target=draw)
        #p.start()


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

        temp = total.copy()

        '''
        temp = total.copy()
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


        '''
        SIZE = 13
        plt.rc('font', size=SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SIZE)    # legend fontsize
        plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title
        '''


        size = 15
        axis_size = 17
        im = ax.imshow(temp, aspect='auto', extent=(0, 2, 0, 0.5), interpolation='gaussian', cmap=cm.inferno_r)
        #cbar = ax.figure.colorbar(im, ax=ax, ticks=[])
        #cbar.ax.tick_params(labelsize=size)
        #cbar.ax.set_ylabel('$\\langle H_P \\rangle$', rotation=0, va="bottom")
        #cbar.ax.get_yaxis().labelpad = 28
        #cbar.ax.set_ylabel('$\\langle H_P \\rangle$', rotation=0, fontsize=axis_size)

        plt.gca().set_xlabel('$\\gamma\,/\,\pi$', fontsize=axis_size)
        plt.gca().set_ylabel('$\\beta\,/\,\pi$', fontsize=axis_size)

        plt.xticks([0, 0.5, 1, 1.5, 2], size=size)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], size=size)

        #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        #plt.gca().set_xlim([0.8,6.2])
        #plt.gca().set_ylim([1, 1.10])


        extent = im.get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1)
        #plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(len(G.nodes)) + ', k=' + str(k) + \
                  #', p=' + str(1) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps) + ', gi=' + str(gi))
        plt.tight_layout()
        #plt.show()
        plt.savefig('random/' + str(num_nodes) + '-' + str(random.randint(1, 10000)) + '.svg')
        #if flag == 0: plt.savefig('figures/p/' + str(gi) + '.png')
        #else: plt.savefig('figures/e/' + str(gi) + '.png')

if __name__ == '__main__':
    temp_map()
