import pandas as pd
import networkx as nx
import ast
import paths
from src.utils.TM_dataset import rating_of_movie, in_top_genres
from src.utils.mathematical import get_all_pairs

pd.set_option('display.max_colwidth', None)
__credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv', usecols=['id', 'cast'])
__movies = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', usecols=['id', 'vote_average'])


def parse_movie_cast(cast, actor_depth):
    movie_cast = []
    res = ast.literal_eval(cast)
    for j in range(min(len(res), actor_depth)):
        movie_cast.append(res[j]['id'])
    return movie_cast


def get_network(actor_depth, coacting_count_threshold):
    """
    generating the undirected actors graph
    each edge has three values:
    {count: number of common movies, rating_sum: sum of common movies' normalized vote averages,
    weight=rating_sum/common_count}
    the edge's weight would be in the range [0, 1]

    :param actor_depth: only the first actor_depth actors are considered in each movie
    :param coacting_count_threshold: if two actors have at least coacting_count_threshold common movies they would have an edge
    :return:
    """
    actors_graph = nx.Graph()
    for i in range(len(__credits)):
        movie_id = __credits['id'][i]
        # checking primary condition: only movies with ratings and top genres
        if not in_top_genres(movie_id):
            continue
        try:
            movie_rating = rating_of_movie(movie_id)
        except Exception:  # no rating has been found for the movie
            continue

        movie_cast = parse_movie_cast(['cast'][i], actor_depth)

        # self-connection edges
        for actor_id in movie_cast:
            if actors_graph.has_edge(actor_id, actor_id):
                common_count = actors_graph[actor_id][actor_id]['count'] + 1
                rating_sum = actors_graph[actor_id][actor_id]['rating_sum'] + movie_rating
            else:
                common_count = 1
                rating_sum = movie_rating

            actors_graph.add_edge(actor_id, actor_id,
                                  count=common_count, rating_sum=rating_sum, weight=rating_sum / common_count)

        edges = get_all_pairs(movie_cast)  # extracting subsets with length equals to 2

        for k in range(len(edges)):
            if actors_graph.has_edge(edges[k][0], edges[k][1]):
                common_count = actors_graph[edges[k][0]][edges[k][1]]['count'] + 1
                rating_sum = actors_graph[edges[k][0]][edges[k][1]]['rating_sum'] + movie_rating
            else:
                common_count = 1
                rating_sum = movie_rating

            actors_graph.add_edge(edges[k][0], edges[k][1],
                                  count=common_count, rating_sum=rating_sum, weight=rating_sum / common_count)

    selected = [(u, v, d) for (u, v, d) in actors_graph.edges(data=True) if d['count'] >= coacting_count_threshold]

    final_graph = nx.Graph()
    final_graph.add_edges_from(selected)

    # edges_list = []
    # for u, v, d in final_graph.edges(data=True):
    #     edges_list.append((u, v, d))

    print('Actors network has been created with {} nodes and {} edges'.format(
        final_graph.number_of_nodes(), final_graph.number_of_edges()))

    return final_graph
