import paths
import pickle
import networkx as nx
from src.modelling.actors_network import get_network
from src.utils.TM_dataset import actors_rating_genre_based


def get_actors_network_features(actor_depth=5, coacting_count_threshold=2):
    actors_adjacency, actors_id = __get_actors_network(actor_depth, coacting_count_threshold)
    actors_feature = __get_actors_features(actors_id, actor_depth, coacting_count_threshold)
    return actors_adjacency, actors_feature, actors_id


def __get_actors_network(actor_depth, coacting_count_threshold):
    graph_file_path = paths.models + 'actors_network/graph_{}_{}.pkl'.format(actor_depth, coacting_count_threshold)
    try:  # loading the graph from disk if it has been saved before
        graph = pickle.load(open(graph_file_path, 'rb'))
        print('Actors graph has been loaded from disk')
    except FileNotFoundError:
        graph = get_network(actor_depth, coacting_count_threshold)
        pickle.dump(graph, open(graph_file_path, 'wb'))

    return nx.adjacency_matrix(graph).toarray(), list(graph.nodes)


def __get_actors_features(actors_id, actor_depth, coacting_count_threshold):
    features_file_path = paths.models + 'actors_network/features_{}_{}.pkl'.format(actor_depth, coacting_count_threshold)
    try:  # loading the node features from disk if they have been saved before
        actors_feature = pickle.load(open(features_file_path, 'rb'))
        print('Actors features have been loaded from disk')
    except FileNotFoundError:
        actors_feature = actors_rating_genre_based(actors_id)
        pickle.dump(actors_feature, open(features_file_path, 'wb'))

    return actors_feature
