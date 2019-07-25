import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

def read_data():
    for gi in range(1, 995):
        path = 'data/' + str(gi) + '.carlo'
        data = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except Exception as e:
                    print(e)
            if data:
                plt.errorbar(data[0], data[1], yerr=data[3], fmt='-o', label='gi=' + str(data[4]))

    #plt.legend()
    plt.gca().set_ylim([0.75, 0.92])
    plt.xticks(np.arange(1, 21, step=1))
    plt.xlabel('$p$ rounds')
    plt.ylabel('Approximation ratio')
    #plt.title('Best solution found via Monte Carlo sampling with s=50 samples')
    plt.show()

if __name__ == '__main__':
    #gather_data()
    read_data()
