import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process
import numpy as np


def app():
    global monte
    stds = []
    for i in range(len(monte[0])):
        stds.append(np.std(monte[0][i]))
    plt.errorbar([x+1 for x in range(len(monte[1]))], stds, fmt='--bo', capsize=5)
    #plt.legend(loc=4, fontsize=17)
    plt.gca().set_ylabel('$\sigma$', fontsize=17, labelpad=10, rotation=0)
    plt.gca().set_xlabel('$p$', fontsize=17)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=17)
    plt.yticks([0, 0.01, 0.02, 0.03], size=17)
    plt.yticks(size=17)
    plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    plt.gca().set_ylim([0,0.03])
    #plt.yscale('log')
    #plt.gca().set_ylim([0.89, 1])
    #plt.title('$p=$' + str(p))
    plt.tight_layout()
    plt.show()

#monte = pickle.load(open('data/2019-09-08 08:33:39.470159', 'rb'))
monte = pickle.load(open('data/2019-09-08 08:46:30.740777', 'rb'))

p = Process(target=app)
p.start()

plt.errorbar([x+1 for x in range(len(monte[1]))], monte[1], yerr=monte[2], fmt='--ro', capsize=5)
#plt.legend(loc=4, fontsize=17)
plt.gca().set_ylabel('Approximation ratio', fontsize=17, labelpad=10)
plt.gca().set_xlabel('$p$', fontsize=17)
plt.xticks([x+1 for x in range(len(monte[0]))], size=17)
plt.yticks([0.8,0.85,0.9,0.95,1], size=17)
plt.yticks(size=17)
plt.gca().set_xlim([0.8,len(monte[0])+0.2])
plt.gca().set_ylim([0.8,1])
#plt.yscale('log')
#plt.gca().set_ylim([0.89, 1])
#plt.title('$p=$' + str(p))
plt.tight_layout()
plt.show()
