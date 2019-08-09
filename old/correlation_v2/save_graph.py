import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import pi
import datetime
import numpy as np
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys, os, pickle
sys.path.insert(0, '/home/montypylon/lanl/qaoa-kvertex/mixer-phase')
import common
import dicke_ps_complete
import random

def write_grid(gi, p):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    num_steps = 50
    gamma, beta = 0, 0
    grid, g_list = [], []
    grid_max = 0
    fig, ax = plt.subplots()

    print('0/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))
    for i in range(0, num_steps):
        for j in range(0, num_steps):
            state = dicke_ps_complete.qaoa([G, C, M, k, p], [gamma, beta])
            exp = common.expectation(G, state)
            g_list.append(exp)
            if grid_max < exp:
                grid_max = exp
            #print('g: ' + str(gamma) + ', b: ' + str(beta) + ', exp: ' + str(exp))
            gamma += pi/(2*(num_steps-1))
        beta += pi/(num_steps-1)
        gamma = 0
        grid.append(g_list)
        g_list = []
        print(str(i+1) + '/' + str(num_steps) + '\t' + str(datetime.datetime.now().time()))

    grid = list(reversed(grid))
    print('-------------- max grid <C>: ' + str(grid_max))

    im = ax.imshow(grid, aspect='auto', extent=(0, pi/2, 0, pi), interpolation='gaussian', cmap=cm.inferno_r)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('$\\langle C \\rangle$', rotation=-90, va="bottom")

    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\beta$')
    plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(len(G.nodes)) + ', k=' + str(k) + \
              ', p=' + str(p) + ', grid_size=' + str(num_steps) + 'x' + str(num_steps) + ', gi=' + str(gi))

    #plt.scatter(opt[0], opt[1], s=50, c='yellow', marker='o')

    folder = 'grid/'
    if not os.path.exists(folder): os.mkdir(folder)
    pickle.dump([[len(G.nodes), k, p, num_steps, gi], grid], open(folder + str(gi) + '.grid', 'wb'))
    plt.show()

def load_grid(gi):
    path = 'grid/' + str(gi) + '.grid'
    data = None
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(e)
    info, grid = data[0], data[1]

    fig, ax = plt.subplots()
    im = ax.imshow(grid, aspect='auto', extent=(0, pi/2, 0, pi), interpolation='gaussian', cmap=cm.inferno_r)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('$\\langle C \\rangle$', rotation=-90, va="bottom")
    plt.xlabel('$\\gamma$')
    plt.ylabel('$\\beta$')
    plt.title('$\\beta \\ vs \\ \\gamma$\nn=' + str(info[0]) + ', k=' + str(info[1]) + \
              ', p=' + str(info[2]) + ', grid_size=' + str(info[3]) + 'x' + str(info[3]) + ', gi=' + str(info[4]))
    plt.show()

def LHS(p, num_samples, lower_g, upper_g, lower_b, upper_b):
    range_g = upper_g - lower_g
    range_b = upper_b - lower_b
    g = [(lower_g + (range_g/num_samples)*(i + 0.5)) for i in range(num_samples)]
    b = [(lower_b + (range_b/num_samples)*(i + 0.5)) for i in range(num_samples)]
    valid_g = [[i for i in range(num_samples)] for j in range(p)]
    valid_b = [[i for i in range(num_samples)] for j in range(p)]
    sample_g = [[] for j in range(num_samples)]
    sample_b = [[] for j in range(num_samples)]
    for j in range(p):
        for i in range(num_samples):
            vg = random.randint(0, len(valid_g[j])-1)
            vb = random.randint(0, len(valid_b[j])-1)
            sample_g[i].append(g[valid_g[j][vg]])
            sample_b[i].append(b[valid_b[j][vb]])
            del valid_g[j][vg]
            del valid_b[j][vb]
    return sample_g, sample_b

def get_opt(gi, p):
    G = nx.read_gpickle('../benchmarks/atlas/' + str(gi) + '.gpickle')
    C = common.create_C(G)
    M = common.create_complete_M(len(G.nodes))
    k = int(len(G.nodes)/2)

    num_steps = 0
    num_samples = 50

    lower_g, upper_g = 1.1, 1.55
    lower_b, upper_b = 1.5, 1.8
    #lower_g, upper_g = 0, pi/2
    #lower_b, upper_b = 0, pi

    init = [[lower_g, upper_g] if i < p else [lower_b, upper_b] for i in range(2*p)]
    sample_g, sample_b = LHS(p, num_samples, lower_g, upper_g, lower_b, upper_b)

    eps = 0.001
    top_exps, top_angles = -1, []

    for i in range(num_samples):
        print('i: ' + str(i))
        print('top_exps: ' + str(top_exps))
        print('top_angles: ' + str(top_angles))

        kwargs = {'method': 'L-BFGS-B', 'args': (G, C, M, k, p), 'bounds': init}
        optimal = basinhopping(dicke_ps_complete.opt, [sample_g[i], sample_b[i]], minimizer_kwargs=kwargs, niter=num_steps, disp=False)
        #optimal = minimize(dicke_ps_complete.opt, np.array([sample_g[i], sample_b[i]]), method='Nelder-Mead', tol=0.01, args=(G, C, M, k, p))

        opt_exp = -optimal.fun
        opt_angles = list(optimal.x)

        if not top_angles:
            top_exps = opt_exp
            top_angles.append(opt_angles)
        elif abs(opt_exp - top_exps) < eps:
            # same value, check if angles are eps similar
            flag = False
            for j in range(len(top_angles)):
                t = 0
                for l in range(len(opt_angles)):
                    t += abs(top_angles[j][l] - opt_angles[l])
                if t < eps:
                    flag = True
                    break
            if not flag:
                top_angles.append(opt_angles)
        elif opt_exp > (top_exps + eps):
            # new maximum
            top_exps, top_angles = opt_exp, []
            top_angles.append(opt_angles)

    print('----------------------')
    print('top_exps: ' + str(top_exps))
    print('top_angles: ' + str(top_angles))

    return top_exps, top_angles

if __name__ == '__main__':
    gi = 91
    p = 7
    #write_grid(gi, p)
    load_grid(gi)

    #opt, angles = get_opt(gi, p)
    #print(opt)

    #for i in range(len(angles)):
    #    plt.scatter(angles[i][0], angles[i][1], s=50, c='yellow', marker='o')

    #plt.show()
    print('Done')
