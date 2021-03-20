import pandas as pd
import networkx as nx
import ast
from itertools import combinations

import paths
from src.utils.TM_dataset import rating_of_movie

pd.set_option('display.max_colwidth', None)
__credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv')


def get_network(actor_depth=5, coacting_count_threshold=2):
    """
    each edge has three values:
    {count: number of common movies, rating_sum: sum of common movies' ratings, weight=rating_sum/common_count/5}
    the edge's weight would be in the range [0, 1]

    :param actor_depth: only the first actor_depth actors are considered in each movie
    :param coacting_count_threshold: if two actors have at least coacting_count_threshold common movies they would have an edge
    :return:
    """
    sample_size = 1000  # TODO: it's for test, remove it
    sample_size = len(__credits)  # TODO

    actors_graph = nx.Graph()
    for i in range(sample_size):
        cast = __credits['cast'][i]
        res = ast.literal_eval(cast)
        movie_cast = []
        movie_id = __credits['id'][i]
        movie_rating = rating_of_movie(movie_id)
        for j in range(min(len(res), actor_depth)):
            movie_cast.append(res[j]['id'])

        edges = list(combinations(movie_cast, 2))  # extracting subsets with length equal to 2

        for k in range(len(edges)):
            if actors_graph.has_edge(edges[k][0], edges[k][1]):
                common_count = actors_graph[edges[k][0]][edges[k][1]]['count'] + 1
                rating_sum = actors_graph[edges[k][0]][edges[k][1]]['rating_sum'] + movie_rating
            else:
                common_count = 1
                rating_sum = movie_rating

            # update the edge's data
            actors_graph.add_edge(edges[k][0], edges[k][1],
                                  count=common_count, rating_sum=rating_sum, weight=rating_sum/common_count/5)

    selected = [(u, v, d) for (u, v, d) in actors_graph.edges(data=True) if d['count'] >= coacting_count_threshold]

    final_graph = nx.Graph()
    final_graph.add_edges_from(selected)

    # edges_list = []
    # for u, v, d in final_graph.edges(data=True):
    #     edges_list.append((u, v, d))
    return nx.adjacency_matrix(final_graph), actors_graph.nodes
