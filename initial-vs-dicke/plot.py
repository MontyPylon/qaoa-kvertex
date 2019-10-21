import matplotlib.pyplot as plt
import pickle
from multiprocessing import Process
import numpy as np
import os
import sys

font_size = 20

def complete():
    global dc_path, kc_path, graph_seed
    dc, kc = None, None
    if os.path.exists(dc_path): dc = pickle.load(open(dc_path + str(graph_seed), 'rb'))
    if os.path.exists(kc_path): kc = pickle.load(open(kc_path + str(graph_seed), 'rb'))
    # dc, kc = [all_samples, max_exps, errors, num_cpus]
    if dc is None and kc is None: return

    if dc is not None:
        plt.plot([x+1 for x in range(len(dc[1]))], dc[1], '--bo', label='Dicke & Complete')
    if kc is not None:
        err = kc[2]
        plt.errorbar([x+1 for x in range(len(kc[1]))], kc[1], yerr=err, fmt='--ro', capsize=5, label='k-state & Complete')

    # y-axis
    plt.gca().set_ylabel('Approximation ratio', fontsize=font_size, labelpad=15)
    #plt.yticks([0.02,0.025,0.03,0.035,0.04,0.045,0.05], size=font_size)
    #plt.gca().set_ylim([0.8, 1.02])
    plt.yticks(size=font_size)
    # x-axis
    plt.gca().set_xlabel('$p$', fontsize=font_size)
    plt.xticks([x+1 for x in range(len(dc[1]))], size=font_size)
    plt.gca().set_xlim([0.8,len(dc[1])+0.2])
    plt.title('Complete', fontsize=font_size)
    #plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.show()

def ring():
    global dr_path, kr_path, graph_seed
    dr, kr = None, None
    if os.path.exists(dr_path): dr = pickle.load(open(dr_path + str(graph_seed), 'rb'))
    if os.path.exists(kr_path): kr = pickle.load(open(kr_path + str(graph_seed), 'rb'))
    # dr, kr = [all_samples, max_exps, errors, num_cpus]
    if dr is None and kr is None: return

    if dr is not None:
        plt.plot([x+1 for x in range(len(dr[1]))], dr[1], '--bo', label='Dicke & Ring')
    if kr is not None:
        err = kr[2]
        plt.errorbar([x+1 for x in range(len(kr[1]))], kr[1], yerr=err, fmt='--ro', capsize=5, label='k-state & Ring')

    # y-axis
    plt.gca().set_ylabel('Approximation ratio', fontsize=font_size, labelpad=15)
    #plt.yticks([0.02,0.025,0.03,0.035,0.04,0.045,0.05], size=font_size)
    #plt.gca().set_ylim([0.8, 1.02])
    plt.yticks(size=font_size)
    # x-axis
    plt.gca().set_xlabel('$p$', fontsize=font_size)
    plt.xticks([x+1 for x in range(len(dr[1]))], size=font_size)
    plt.gca().set_xlim([0.8,len(dr[1])+0.2])
    plt.title('Ring', fontsize=font_size)
    #plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.show()

# data = [all_samples, max_exps, errors, num_cpus]
# all_samples contains p arrays of size num_cpus, containing best found exp
# max_exps = [0.85, 0.87, 0.94, 0.96, 0.98, ...] of size p
# errors are the error in max_exps with sample size num_cpus (the var size)
dc_path = 'dicke_complete/'
dr_path = 'dicke_ring/'
kc_path = 'k_complete/'
kr_path = 'k_ring/'

# arguments: [(int) graph_seed]
graph_seed = int(sys.argv[1])

pc = Process(target=complete)
pc.start()
pr = Process(target=ring)
pr.start()
