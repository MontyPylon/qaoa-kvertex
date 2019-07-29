import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot():
    complete = pickle.load(open('data/91.complete', 'rb'))
    print(complete)
    # [gi, [1,...,p], [avg], [std], [error], num_samples, num_cores]

    plt.errorbar(complete[1], complete[2], yerr=complete[4], fmt='-o', label='complete')

    plt.legend()
    #plt.gca().set_ylim([0.80, 1])
    plt.xticks(np.arange(1, len(complete[1]), step=1))
    plt.xlabel('$p$ rounds')
    plt.ylabel('Approximation ratio')
    #plt.title('Best solution found via Monte Carlo sampling with s=50 samples')
    plt.show()

if __name__ == '__main__':
    plot()
