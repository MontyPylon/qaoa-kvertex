import networkx as nx
import multiprocessing
import time
import pickle
from ring_ps_ring import ring_ps_ring

def worker(gi, dict_exp):
    G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
    if k == 0: k = 1
    key = tuple([gi, k, p])
    n = 25
    exp, angles = ring_ps_ring(G, k, p, n)
    dict_exp[key] = exp
    #dict_angles[key] = angles

if __name__ == '__mains__':
    manager = multiprocessing.Manager()
    dict_exp = manager.dict()
    #dict_angles = manager.dict()
    jobs = []
    for gi in range(1, 10):
        p = multiprocessing.Process(target=worker, args=(gi, dict_exp))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(dict_exp.values())

def basic_func(x):
    if x == 0:
        return 'zero'
    elif x%2 == 0:
        return 'even'
    else:
        return 'odd'

def multiprocessing_func(gi):
    G = nx.read_gpickle('/home/montypylon/lanl/qaoa-kvertex/hpc/atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
    if k == 0: k = 1
    n = 3
    p = 1
    key = tuple([gi, k])
    print('starting ' + str(gi))
    exp, angles = ring_ps_ring(G, k, p, n)
    print('finished ' + str(gi))
    dict_exp[key] = exp

if __name__ == '__main__':
    starttime = time.time()

    manager = multiprocessing.Manager()
    dict_exp = manager.dict()

    pool = multiprocessing.Pool()
    pool.map(multiprocessing_func, range(1,4))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))
    pickle.dump(dict_exp, open('data/ring_ps_ring.exp', 'wb'))
    print(dict_exp)
