import matplotlib.pyplot as plt
import networkx as nx


def draw_bipartite_graph(graph, first_node_set):
    pos = nx.bipartite_layout(graph, first_node_set)
    # get adjacency matrix
    A = nx.adjacency_matrix(graph)
    A = A.toarray()
    # plot adjacency matrix
    plt.imshow(A, cmap='Greys')
    plt.show()
    # plot graph visualisation
    nx.draw(graph, pos, with_labels=False)
    plt.show()
