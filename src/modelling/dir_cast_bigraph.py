import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.TM_dataset import rating_of_movie


def build_graph(credits_df):
    """
    to build a weighted bi-graph of directors and casts
    each weight is equal to average of the ratings of movies the cast and crew have involved in
    :return: a bi-graph
    """
    director_set, cast_set = set(), set()
    director_name, cast_name = {}, {}
    edges_count, edges_sum_rate = defaultdict(int), defaultdict(float)

    for _, row in credits_df.iterrows():
        crews = eval(row.crew)
        try:
            director = (next(crew for crew in crews if crew['job'] == 'Director'))
            director_id = 'dir_' + str(director['id'])
            director_set.add(director_id)
            director_name[director_id] = director['name']

        except StopIteration:  # when it cannot find the director in the crew list
            continue

        rating = rating_of_movie(row.id)
        if np.isnan(rating):  # TODO: or should I create the edge with some weight?
            continue

        casts = eval(row.cast)
        for cast in casts:
            cast_id = 'cast_' + str(cast['id'])
            cast_set.add(cast_id)
            cast_name[cast_id] = cast['name']

            edges_count[(director_id, cast_id)] += 1
            edges_sum_rate[(director_id, cast_id)] += rating

    # statistics about the movies dataset
    # print(len(director_set))  # 17698
    # print(len(cast_set))  # 205467
    # print(len(edges))  # 513485

    dir_cast_graph = nx.Graph()
    dir_cast_graph.add_nodes_from(director_set, bipartite=0)
    dir_cast_graph.add_nodes_from(cast_set, bipartite=1)

    for edge, count in edges_count.items():
        sum_rates = edges_sum_rate[edge]
        weighted_edge = edge + (sum_rates/count,)
        dir_cast_graph.add_weighted_edges_from([weighted_edge])

    return dir_cast_graph, director_set, cast_set, director_name, cast_name


def predict_links(dir_cast_graph, director_set, cast_set, potentiality_threshold=0):
    cast_list = list(cast_set)  # mapping from cast_index to cast_id
    director_list = list(director_set)  # mapping from director_index to director_id

    adjacency_matrix = np.zeros([len(director_list), len(cast_list)])  # index = director_index, cast_index
    # if there is not any edge between a director and a cast, the related value equals to zero

    for director_id in director_set:
        director_index = director_list.index(director_id)
        for edge in dir_cast_graph.edges(director_id):
            cast_id = edge[1]
            cast_index = cast_list.index(cast_id)
            weight = dir_cast_graph.get_edge_data(*edge)['weight']
            adjacency_matrix[director_index, cast_index] = weight

    similarities = cosine_similarity(adjacency_matrix, adjacency_matrix)
    print('--similarities are calculated between directors')

    director_cast_potentialities = []  # each element is a list [director_id, cast_id, potentiality]

    for director_idx_i in range(len(director_list)):

        sum_cast_potentiality = np.array([.0] * len(cast_list))
        for director_idx_j in range(len(director_list)):
            if director_idx_i == director_idx_j:
                continue
            single_cast_potentiality = adjacency_matrix[director_idx_j] - adjacency_matrix[director_idx_i]
            # shows all casts' potentialities to be well-matched with director i
            single_cast_potentiality[single_cast_potentiality < 0] = 0

            sum_cast_potentiality += (single_cast_potentiality * similarities[director_idx_i, director_idx_j])

        cast_potentiality_i = sum_cast_potentiality / (len(director_list) - 1)
        for cast_idx in range(len(cast_list)):
            director_id = director_list[director_idx_i]
            cast_id = cast_list[cast_idx]
            potentiality = cast_potentiality_i[cast_idx]
            if potentiality > potentiality_threshold:
                director_cast_potentialities.append([director_id, cast_id, potentiality])

    return director_cast_potentialities
