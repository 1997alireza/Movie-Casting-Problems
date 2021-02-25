import paths
import pandas as pd
import networkx as nx


def build_graph():
    """

    :return: the bi-graph of directors and casts
    """
    the_movies_df = pd.read_csv(paths.the_movies_dataset + 'credits.csv',
                                usecols=['cast', 'crew'])

    director_set, cast_set = set(), set()
    director_name, cast_name = {}, {}
    edges = set()

    for _, row in the_movies_df.iterrows():
        crews = eval(row.crew)
        try:
            director = (next(crew for crew in crews if crew['job'] == 'Director'))
            director_id = 'dir_' + str(director['id'])
            director_set.add(director_id)
            director_name[director_id] = director['name']

        except StopIteration:  # when it cannot find the director in the crew list
            continue

        casts = eval(row.cast)
        for cast in casts:
            cast_id = 'cast_' + str(cast['id'])
            cast_set.add(cast_id)
            cast_name[cast_id] = cast['name']

            edges.add((director_id, cast_id))

    # print(len(director_set))  # 17698
    # print(len(cast_set))  # 205467
    # print(len(edges))  # 513485

    dir_cast_graph = nx.Graph()
    dir_cast_graph.add_nodes_from(director_set, bipartite=0)
    dir_cast_graph.add_nodes_from(cast_set, bipartite=1)
    dir_cast_graph.add_edges_from(edges)

    # print(bipartite.is_bipartite(dir_cast_graph))  # True

    return dir_cast_graph




