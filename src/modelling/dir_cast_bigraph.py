import paths
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def build_graph():
    """

    :return: the bi-graph of directors and casts
    """
    the_movies_df = pd.read_csv(paths.the_movies_dataset + 'credits.csv',
                                usecols=['cast', 'crew'])

    # the_movies_df = the_movies_df[:100]  # TODO: remove

    director_set, cast_set = set(), set()
    director_name, cast_name = {}, {}
    edges_count = defaultdict(int)

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

            edges_count[(director_id, cast_id)] += 1

    # print(len(director_set))  # 17698
    # print(len(cast_set))  # 205467
    # print(len(edges))  # 513485

    dir_cast_graph = nx.Graph()
    dir_cast_graph.add_nodes_from(director_set, bipartite=0)
    dir_cast_graph.add_nodes_from(cast_set, bipartite=1)

    for edge, count in edges_count.items():
        weighted_edge = edge + (count,)
        dir_cast_graph.add_weighted_edges_from([weighted_edge])

    return dir_cast_graph


# if __name__ == '__main__':
#     dir_cast_graph = build_graph()
#     nx.draw(dir_cast_graph, with_labels=True)
#     plt.savefig("graph.png")
