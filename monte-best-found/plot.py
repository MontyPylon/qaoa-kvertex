import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process
import numpy as np


def app():
    global monte

    opt404 = [8.491516852755565/9, 8.771279196409646/9, 8.907085224838939/9, 8.95/9, 8.97/9, 8.985/9, 8.99/9, 1, 1]
    plt.plot([x+1 for x in range(len(opt404))], opt404, '-o')

    plt.plot([x+1 for x in range(len(monte[0]))], monte[2], '-o')
    plt.gca().set_ylabel('Approximation ratio', fontsize=17, labelpad=10)
    plt.gca().set_xlabel('$p$', fontsize=17)
    plt.yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1], size=17)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=17)
    plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    plt.gca().set_ylim([0.75, 1.01])
    plt.tight_layout()
    plt.show()

monte = pickle.load(open('data/2019-09-08 08:27:11.670979', 'rb'))

#p = Process(target=app)
#p.start()

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
