import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot():
    total_c = []
    total_r = []
    for gi in range(330, 340):
        path = 'complete/' + str(gi) + '.mpi'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data: total_c.append(data)

        path = 'ring/' + str(gi) + '.mpi'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data: total_r.append(data)


    ratio = []
    for i in range(len(total_c[0])):
        local = []
        for j in range(len(total_c)):
            local.append(total_c[j][i]/total_r[j][i])
        ratio.append(local)

    avg, std = [], []
    for i in range(len(total_c[0])):
        avg.append(np.average(ratio[i]))
        std.append(np.std(ratio[i]))

    error = [i*2.576/np.sqrt(len(total_c)) for i in std]

    x = [i+1 for i in range(6)]

    '''
    SIZE = 16
    plt.rc('font', size=SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=12)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title
    '''
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

    plt.plot(x, total_c[0], '-o', color='b', label='$H_K$: complete mixer')
    plt.plot(x, total_r[0], '--v', color='r', label='$H_R$: ring mixer')
    plt.legend(prop={'size': 15})
    plt.gca().set_ylabel('Approximation ratio', fontsize=15, labelpad=10)

    #plt.errorbar(x, avg, yerr=error, fmt='-o', color='k', capsize=10)
    #plt.gca().set_ylim([1, 1.10])
    #plt.xlabel('$p$')
    #plt.ylabel('$\\frac{r_K}{r_R}$', rotation=0, fontsize=20)
    #plt.gca().set_ylabel('$\\frac{r_K}{r_R}$', rotation=0, fontsize=30, labelpad=15)
    plt.gca().set_xlabel('$p$', fontsize=20)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot()
