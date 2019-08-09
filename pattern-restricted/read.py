import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from math import pi

def read_data():
    p = 4
    lines_g = []
    lines_b = []

    for gi in range(160, 955):
        path = 'good/' + str(gi) + '.angles'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data:
                #print('\n' + str(gi))
                for i in range(len(data[0])):
                    #print(data[0][i])
                    for j in range(len(data[1][i])):
                        for k in range(len(data[1][i][j])):
                            if k <= i: data[1][i][j][k] = data[1][i][j][k]#pi/2 - data[1][i][j][k]
                            else: data[1][i][j][k] = pi - data[1][i][j][k]
                        if i+1 == p:
                            lines_g.append(data[1][i][j][:p])
                            lines_b.append(data[1][i][j][p:])
                            if lines_b[len(lines_b)-1][0] > 0.2:
                                print(gi)
                        #print('\t' + str(data[1][i][j]))
                #print('\n')

    print(lines_g)
    print(lines_b)

    SIZE = 20
    plt.rc('font', size=SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title


    '''
    for i in range(len(lines_g)):
        plt.plot([x+1 for x in range(p)], lines_g[i], '-o', label=str(i))
    plt.xlabel('$\\gamma_i$')
    plt.ylabel('Value of $\\gamma_i$')
    '''
    for i in range(len(lines_b)):
        plt.plot([x+1 for x in range(p)], lines_b[i], '-o')
    plt.xlabel('$\\beta_i$')
    plt.ylabel('Value of $\\beta_i$')


    plt.xticks([x+1 for x in range(p)])
    #plt.legend()
    plt.gca().set_ylim([0, 0.16])
    #plt.xticks(np.arange(1, , step=1))
    plt.tight_layout()
    plt.title('$p=$' + str(p))
    plt.show()

if __name__ == '__main__':
    read_data()
