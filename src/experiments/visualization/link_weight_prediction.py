import pickle
import paths
import networkx as nx
import matplotlib.pyplot as plt
from src.processing.GAE_on_actors import LinkWeightPredictor
from src.utils.mathematical import get_all_pairs


def extract_candidate_subgraph(graph, size_range=(10, 20), weight_range=(.45, .75)):
    for c in nx.connected_components(graph):
        if size_range[0] < len(c) < size_range[1]:
            subgraph = graph.subgraph(c)
            weights = [w for u, v, w in subgraph.edges.data("weight")]
            if not(min(weights) <= weight_range[0] and max(weights) >= weight_range[1]):
                continue
            print(len(c), min(weights), max(weights), c)
            return subgraph


def show_graph(graph, pos, new_color=False):
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    for e in edge_labels:
        edge_labels[e] = int(edge_labels[e] * 1e2) / 1e2
    nx.draw_networkx_edge_labels(graph,
                                 edge_labels=edge_labels,
                                 pos=pos)
    if new_color:
        nx.draw(graph, pos, node_color='red')
    else:
        nx.draw(graph, pos)
    plt.show()


def predict_plot_graphs(predictor, graph):
    original_edges, predicted_edges = [], []
    for pair in get_all_pairs(graph.nodes):
        try:
            original_weight = graph.get_edge_data(*pair)['weight']
            original_edges.append((*pair, {'weight': original_weight}))
        except Exception:  # when the edge is not presented in the graph
            pass
        weight = predictor.predict(pair[0], pair[1])
        predicted_edges.append((*pair, {'weight': weight}))

    pos = nx.spring_layout(graph)

    graph = nx.Graph()
    graph.add_edges_from(original_edges)  # removing loop edges from the original graph

    predicted_graph = nx.Graph()
    predicted_graph.add_edges_from(predicted_edges)

    show_graph(graph, pos)
    show_graph(predicted_graph, pos, new_color=True)


if __name__ == '__main__':
    graph_file_path = paths.models + 'actors_network/graph_{}_{}.pkl'.format(5, 2)
    graph = pickle.load(open(graph_file_path, 'rb'))
    predictor = LinkWeightPredictor()

    subgraph = extract_candidate_subgraph(graph)
    # a large subgraph: nodes {87676, 13506, 65476, 143716, 92710, 220295, 13514, 116202, 44366, 82420, 45749, 82422, 82423, 79448, 45754, 43708, 220990}: weights in range 0.45 0.752
    predict_plot_graphs(predictor, subgraph)

    subgraph = extract_candidate_subgraph(graph, size_range=(3, 15), weight_range=(.5, .8))
    # a small subgraph: nodes {27392, 129571, 120656, 126975}: weights in range [0.495, 0.865]
    predict_plot_graphs(predictor, subgraph)

    # the graphs' plot are located in /docs/plots/actors_subgraph/
