import networkx as nx
import matplotlib.pyplot as plt
from dicke_hprime_ring import dicke_hprime_ring
from dicke_hprime_parityring import dicke_hprime_parityring
from ring_hprime_ring import ring_hprime_ring
from pring_ps_pring import pring_ps_pring
from brute_force import brute_force
import datetime
import os

def get_exp(G, gi, k, method, method_string):
    exp = None
    new_dict = None
    #path = 'angles/' + method_string + '.angles'
    path = 'angles/' + method_string + '.exp'

    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                exp_dic = pickle.load(f)
                key = [gi, k]
                if key in exp_dic:
                    exp = exp_dic[key]
            except: 
                print('Error opening dictionary for ' + method_string)
    else:
        print('File does not exist for ' + method_string)
        f = open(path, 'wb')


    #if exp is None:
    #    exp, angles, ab = method(G, k)

    return 0
    #return exp

def write_angles(gi):
    y1 = []
    value = brute_force(G, k)
    y2.append(ring_hprime_ring(G, k, p, n))
    y3.append(dicke_hprime_ring(G, k, p, n))

def compare():
    # min: 1, max: 995
    start = 31
    end = 32
    p = 1
    n = 3
    x = []
    y1 = []
    
    print(str(start) + '/' + str(end) + '\t' + str(datetime.datetime.now().time()))
    for gi in range(start, end):
        G = nx.read_gpickle('benchmarks/atlas/' + str(gi) + '.gpickle')
        #k = int(len(G.nodes)/2)
        k = 1
        x.append('gi=' + str(gi) + ',n=' + str(len(G.nodes)) + ',k=' + str(k))
        #exps = write_angles(gi)

        y1.append(get_exp(G, gi, k, brute_force, 'brute_force')) 

        print(str(gi+1) + '/' + str(end) + '\t' + str(datetime.datetime.now().time()))

    plt.plot(x, y1, '-bo', label='brute_force')
    #plt.plot(x, y2, '-go', label='ring_hprime_ring')
    #plt.plot(x, y3, '-ro', label='dicke_hprime_ring')
    plt.legend()

    plt.xlabel('Graph atlas index')
    plt.ylabel('<C>: expectation of number of edges touched')
    plt.title('<C> vs. graph atlas\np=' + str(p) + ', grid_size=' + str(n) + 'x' + str(n))
    plt.show()
        
if __name__ == '__main__':
    #generate_graphs()
    compare()
