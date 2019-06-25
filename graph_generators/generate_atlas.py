import networkx as nx
import matplotlib.pyplot as plt

def generate_graphs():
    # creates all graphs up to 7 nodes (up to isomorphism)
    # max valid number is 1252
    graph_num = 0
    for i in range(2, 1253):
        G = nx.graph_atlas(i)
        if not nx.is_connected(G):
            continue
        graph_num += 1
        nx.write_gpickle(G, '../mixer-phase/benchmarks/atlas/' + str(graph_num) + '.gpickle')

def read_graphs():
    for i in range(1, 996):
        G = nx.read_gpickle('../mixer-phase/benchmarks/atlas/' + str(i) + '.gpickle')
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

def draw_graphs():
    for i in range(1, 996):
        G = nx.read_gpickle('../mixer-phase/benchmarks/atlas/' + str(i) + '.gpickle')
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.savefig('../mixer-phase/benchmarks/atlas_images/' + str(i) + '.png')
        plt.clf()

if __name__ == '__main__':
    #generate_graphs()
    #read_graphs()
    draw_graphs()
