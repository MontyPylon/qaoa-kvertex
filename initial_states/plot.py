import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot():
    total_c = []
    total_i = []
    for gi in range(160, 955):
        path = 'complete/' + str(gi) + '.mpi'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data: total_c.append(data)

        path = 'init-complete/' + str(gi) + '.mpi'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data: total_i.append(data)

    x = [i for i in range(6)]
    for i in range(len(total_c)):
        #plt.plot(x, total_c[i], '-o')
        plt.plot(x, total_i[i][0], '-o')
    print(total_c)
    print(total_i)
    plt.show()

def old_plot():
    data = pickle.load(open('data/91.mpi-k', 'rb'))
    # [gi, [1,...,p], [avg], [std], [error], num_samples, num_cores]
    print(data)

    dicke = [8.503162903355406, 8.800506134417677, 8.952913838859514, 8.984216297214182, 8.993237586035384, 8.998175490445554]#, 8.99978203457083]
    dicke = [i/9 for i in dicke]
    #print(dicke)
    plt.plot([1,2,3,4,5,6], dicke, '-o', label='dicke')
    plt.errorbar(data[1], data[2], yerr=data[4], fmt='-o', label='k-state')

    plt.legend()
    #plt.gca().set_ylim([0.80, 1])
    plt.xticks(np.arange(1, 6, step=1))
    plt.xlabel('$p$ rounds')
    plt.ylabel('Approximation ratio')
    #plt.title('Best solution found via Monte Carlo sampling with s=50 samples')
    plt.show()

if __name__ == '__main__':
    plot()
