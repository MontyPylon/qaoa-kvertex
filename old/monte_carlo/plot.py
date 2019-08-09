import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def read_data():
    # [gi, [1,...,p], [exp], [std], [error], s_per_rank, num_cpus]
    for gi in range(1, 995):
        path = 'data/' + str(gi) + '.mpi-carlo'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data:
                print(data)
                v = ''
                if gi == 910: v = 'p^0'
                if gi == 911: v = 'p^1'
                if gi == 912: v = 'p^2'
                plt.errorbar(data[1], data[2], yerr=data[4], fmt='-o', label=v)

    plt.legend()
    #plt.gca().set_ylim([0.89, 1])
    #plt.xticks(np.arange(1, , step=1))
    plt.xlabel('$p$ rounds')
    plt.ylabel('Approximation ratio')
    plt.title('Monte Carlo sampling with s*p^n samples, with s=100')
    plt.show()

if __name__ == '__main__':
    #gather_data()
    read_data()
