import networkx as nx
import matplotlib.pyplot as plt

def generate_graphs():
    # creates all graphs up to 7 nodes (up to isomorphism)
    # max is 1252
    graph_num = 0
    for i in range(2, 1253):
        G = nx.graph_atlas(i)
        if not nx.is_connected(G):
            continue
        graph_num += 1
        nx.write_edgelist(G, 'graphs/atlas/' + str(graph_num) + '.edgelist')

def read_graphs():
    for i in range(1, 996):
        G = nx.read_edgelist('graphs/atlas/' + str(i) + '.edgelist')
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

def draw_regular():
    n = 10
    d = 3
    while True:
        G = nx.random_regular_graph(d, n, seed=None)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

if __name__ == '__main__':
    #generate_graphs()
    #read_graphs()
    draw_regular()
