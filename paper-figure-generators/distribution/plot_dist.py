import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process
import numpy as np

font_size = 20

def std():
    global monte
    err = [i*0.75 for i in monte[4]]
    plt.errorbar([x+1 for x in range(len(monte[1]))], monte[2], yerr=err, fmt='--ro', capsize=5)
    # y-axis
    plt.gca().set_ylabel('$\sigma$', fontsize=font_size, labelpad=15, rotation=0)
    #plt.yticks([0.02,0.025,0.03,0.035,0.04,0.045,0.05], size=font_size)
    #plt.gca().set_ylim([0.018, 0.052])
    plt.yticks(size=font_size)
    # x-axis
    plt.gca().set_xlabel('$p$', fontsize=font_size)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=font_size)
    plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    plt.tight_layout()
    plt.show()

def avg():
    plt.errorbar([x+1 for x in range(len(monte[1]))], monte[1], yerr=monte[3], fmt='--ko', capsize=5)
    # y-axis
    plt.gca().set_ylabel('Average approximation ratio', fontsize=font_size, labelpad=10)
    #plt.yticks([0.714,0.715,0.716,0.717,0.718,0.719], size=font_size)
    #plt.gca().set_ylim([0.7136, 0.7194])
    plt.yticks(size=font_size)
    # x-axis
    plt.gca().set_xlabel('$p$', fontsize=font_size)
    plt.xticks([x+1 for x in range(len(monte[0]))], size=font_size)
    plt.gca().set_xlim([0.8,len(monte[0])+0.2])
    plt.tight_layout()
    plt.show()

# good std data:
monte = pickle.load(open('dist/2019-09-08 10:33:30.502320', 'rb'))

# graph of size 7, exp decreasing avg and std
#monte = pickle.load(open('dist/2019-10-08 19:30:19.191835', 'rb'))
#monte = pickle.load(open('dist/2019-10-08 19:30:19.191835', 'rb'))


p1 = Process(target=std)
p1.start()
p2 = Process(target=avg)
p2.start()

