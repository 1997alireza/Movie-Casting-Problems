import pandas as pd
import networkx as nx
import ast
from itertools import combinations

import paths

pd.set_option('display.max_colwidth', None)

# credits dataframe including casts and crew
credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv')
# print(credits.head())

# sample cast entry
k = credits['cast'][0]
res = ast.literal_eval(k)
# print(res[0])

# creating graph based on co-acting
MG = nx.MultiGraph()


def rSubset(arr):
    return list(combinations(arr, 2))


for i in range(1000):
    cast = credits['cast'][i]
    res = ast.literal_eval(cast)
    cast_mem = []
    for j in range(len(res)):
        cast_mem.append(res[j]['name'])
    edges = rSubset(cast_mem)
    for k in range(len(edges)):
        MG.add_edge(edges[k][0], edges[k][1])

print('no of nodes: ', len(MG.nodes()))
print('no of edges: ', len(MG.edges()))
conn_comp = list(nx.connected_components(MG))
print('no of connected components:', len(conn_comp))
