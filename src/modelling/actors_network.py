import pandas as pd
import networkx as nx
import ast
from itertools import combinations

import paths
from src.utils.TM_dataset import rating_of_movie

pd.set_option('display.max_colwidth', None)
credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv')
sample_size = 1000


def rSubset(arr):
    return list(combinations(arr, 2))


def get_network(actor_depth, coacting_count):
    actors_graph = nx.Graph()
    for i in range(sample_size):
        cast = credits['cast'][i]
        res = ast.literal_eval(cast)
        movie_cast = []
        movie_id = credits['id'][i]
        movie_rating = rating_of_movie(movie_id)
        for j in range(min(len(res), actor_depth)):
            movie_cast.append(res[j]['id'])
        edges = rSubset(movie_cast)
        for k in range(len(edges)):
            if actors_graph.has_edge(edges[k][0], edges[k][1]):
                common_count = actors_graph[edges[k][0]][edges[k][1]]["weight"] + 1
                rating_sum = actors_graph[edges[k][0]][edges[k][1]]["rating"] + movie_rating
            else:
                common_count = 1
                rating_sum = movie_rating
            actors_graph.add_edge(edges[k][0], edges[k][1],
                                  weight=common_count, rating=rating_sum)
    selected = [(u, v, d) for (u, v, d) in actors_graph.edges(data=True) if d['weight'] >= coacting_count]
    final_graph = nx.Graph()
    final_graph.add_edges_from(selected)
    edges_list = []
    for u, v, d in final_graph.edges(data=True):
        edges_list.append((u, v, d))
    return edges_list


print(get_network(5, 2))

