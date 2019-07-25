import networkx as nx
import multiprocessing
import time
import pickle
from ring_ps_ring import ring_ps_ring
import datetime

def worker(gi, dict_exp, dict_angles):
    G = nx.read_gpickle('/users/jeremycook/atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
    if k == 0: k = 1
    p = 1
    n = 25
    key = tuple([gi, k])
    exp, angles = ring_ps_ring(G, k, p, n)
    dict_exp[key] = exp
    dict_angles[key] = angles
    print('Done: ' + str(gi) + '\t' + str(datetime.datetime.now().time()))

'''
if __name__ == '__main__':
    print('Starting at \t' + str(datetime.datetime.now().time()))
    manager = multiprocessing.Manager()
    dict_exp = manager.dict()
    dict_angles = manager.dict()
    jobs = []
    for gi in range(1, 501):
        p = multiprocessing.Process(target=worker, args=(gi, dict_exp, dict_angles))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    local_exp = {}
    local_exp.update(dict_exp)
    local_angles = {}
    local_angles.update(dict_angles)
    pickle.dump(local_exp, open('/users/jeremycook/data/ring_ps_ring.exp', 'wb'))
    pickle.dump(local_angles, open('/users/jeremycook/data/ring_ps_ring.angles', 'wb'))
    print('Finished at \t' + str(datetime.datetime.now().time()))
'''

def multiprocessing_func(gi):
    print('begin ' + str(gi) + '\t' + str(datetime.datetime.now().time()))
    G = nx.read_gpickle('atlas/' + str(gi) + '.gpickle')
    k = int(len(G.nodes)/2)
    if k == 0: k = 1
    p = 1
    n = 25
    key = tuple([gi, k])
    exp, angles = ring_ps_ring(G, k, p, n)
    dict_exp[key] = exp
    dict_angles[key] = angles
    print('Done: ' + str(gi) + '\t' + str(datetime.datetime.now().time()))

if __name__ == '__main__':
    print('Starting at \t' + str(datetime.datetime.now().time()))

    manager = multiprocessing.Manager()
    dict_exp = manager.dict()
    dict_angles = manager.dict()

    pool = multiprocessing.Pool()
    pool.map(multiprocessing_func, range(1, 11))
    pool.close()

    local_exp = {}
    local_exp.update(dict_exp)
    local_angles = {}
    local_angles.update(dict_angles)
    pickle.dump(local_exp, open('data/ring_ps_ring.exp', 'wb'))
    pickle.dump(local_angles, open('data/ring_ps_ring.angles', 'wb'))
    print('Finished at \t' + str(datetime.datetime.now().time()))
