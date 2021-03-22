import pandas as pd
import networkx as nx
import ast
from itertools import combinations

import paths
from src.utils.TM_dataset import rating_of_movie, in_top_genres

pd.set_option('display.max_colwidth', None)
__credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv', usecols=['id', 'cast'])
__movies = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', usecols=['id', 'vote_average'])


def ratings_stats():
    notrated_count = 0
    rated_count = 0
    for i in range(__movies.shape[0]):
        movie_id = __credits['id'][i]
        try:
            rating = __movies[__movies['id'] == str(movie_id)]
            rated_count += 1
        except Exception:  # no rating has been found for the movie
            notrated_count += 1
            continue
    print("rated: " + str(rated_count))
    print("not rated: " + str(notrated_count))


def get_network(actor_depth=5, coacting_count_threshold=2):
    """
    generating the undirected actors graph
    each edge has three values:
    {count: number of common movies, rating_sum: sum of common movies' ratings, weight=rating_sum/common_count/5}
    the edge's weight would be in the range [0, 1]

    :param actor_depth: only the first actor_depth actors are considered in each movie
    :param coacting_count_threshold: if two actors have at least coacting_count_threshold common movies they would have an edge
    :return:
    """
    # sample_size = 5000  # TODO: it's for test, remove it
    sample_size = len(__credits)  # TODO

    actors_graph = nx.Graph()
    for i in range(sample_size):
        movie_id = __credits['id'][i]
        if not in_top_genres(movie_id):
            continue

        cast = __credits['cast'][i]
        res = ast.literal_eval(cast)
        movie_cast = []

        try:
            movie_rating = rating = __movies[__movies['id'] == str(movie_id)]
        except Exception:  # no rating has been found for the movie
            continue

        for j in range(min(len(res), actor_depth)):
            movie_cast.append(res[j]['id'])

        # self-connection edges
        for actor_id in movie_cast:
            if actors_graph.has_edge(actor_id, actor_id):
                common_count = actors_graph[actor_id][actor_id]['count'] + 1
                rating_sum = actors_graph[actor_id][actor_id]['rating_sum'] + movie_rating
            else:
                common_count = 1
                rating_sum = movie_rating

            actors_graph.add_edge(actor_id, actor_id,
                                  count=common_count, rating_sum=rating_sum, weight=rating_sum / common_count / 5)

        edges = list(combinations(movie_cast, 2))  # extracting subsets with length equals to 2

        for k in range(len(edges)):
            if actors_graph.has_edge(edges[k][0], edges[k][1]):
                common_count = actors_graph[edges[k][0]][edges[k][1]]['count'] + 1
                rating_sum = actors_graph[edges[k][0]][edges[k][1]]['rating_sum'] + movie_rating
            else:
                common_count = 1
                rating_sum = movie_rating

            actors_graph.add_edge(edges[k][0], edges[k][1],
                                  count=common_count, rating_sum=rating_sum, weight=rating_sum / common_count / 5)

    selected = [(u, v, d) for (u, v, d) in actors_graph.edges(data=True) if d['count'] >= coacting_count_threshold]

    final_graph = nx.Graph()
    final_graph.add_edges_from(selected)

    # edges_list = []
    # for u, v, d in final_graph.edges(data=True):
    #     edges_list.append((u, v, d))

    print('Actors network has been created with {} nodes and {} edges.'.format(
        final_graph.number_of_nodes(), final_graph.number_of_edges()))
    return nx.adjacency_matrix(final_graph).toarray(), list(actors_graph.nodes)
