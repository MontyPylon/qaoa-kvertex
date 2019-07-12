import networkx as nx
import numpy as np
from itertools import combinations 
import random

def generate_graph():
    # Pick a random graph from the atlas
    gi = random.randint(2,995)
    print('Graph index: ' + str(gi))
    G = nx.read_gpickle('../mixer-phase/benchmarks/atlas/' + str(gi) + '.gpickle')
    return G, gi

def brute_force(G, k, p, n):
    comb = combinations(G.nodes, k)
    highest = 0
    best_group = []
    for group in list(comb):
        score = 0
        for edge in G.edges:
            for v in group:
                if v == edge[0] or v == edge[1]:
                    score += 1
                    break
        if score > highest:
            highest = score
            best_group = list(group)
    return highest, best_group

if __name__ == '__main__':
    G, gi = generate_graph()
    brute_force(G, int(len(G.nodes)/2), 0, 0)
