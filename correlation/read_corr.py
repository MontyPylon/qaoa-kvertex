import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pickle
import os

def exp_angles(gi, p):
    folder = 'angles/'
    path = folder + 'optimal.exp'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                angle_dict = pickle.load(f)
                key = tuple([gi, p])
                if key in angle_dict:
                    return angle_dict[key]
            except Exception as e:
                print('Error opening dictionary')
                print(e)
    return None

if __name__ == '__main__':
    p = 3
    for gi in range(1, 996):
        G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
        x = [i+1 for i in range(p)]
        angles = exp_angles(gi, p)

        if angles is not None:
            g = [angles[i] for i in range(p)]
            #b = [angles[i+p] for i in range(p)]
            plt.plot(x, g, '-o', label=str(gi))
            #plt.plot(x, b, '-o', label='beta')

    axes = plt.gca()
    axes.set_ylim([0, pi/2])
    plt.xticks(np.arange(1, p+1, step=1))
    plt.legend()
    plt.xlabel('gamma_i')
    plt.ylabel('value of gamma_i')
    plt.title('Gamma vector correlation, p=' + str(p) + ', k=floor(n/2)')
    plt.show()


