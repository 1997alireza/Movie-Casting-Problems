import pandas as pd
import networkx as nx
import ast
from itertools import combinations
import collections
import matplotlib.pyplot as plt
from netwulf import visualize

import paths

pd.set_option('display.max_colwidth', None)


def rSubset(arr):
    return list(combinations(arr, 2))


# credits dataframe including casts and crew
credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv')
# print(credits.head())

# sample cast entry - just warm up!
k = credits['cast'][0]
res = ast.literal_eval(k)
# print(res[0])

# creating graph based on co-acting
MG = nx.Graph()
for i in range(50):
    cast = credits['cast'][i]
    res = ast.literal_eval(cast)
    cast_mem = []
    for j in range(len(res)):
        cast_mem.append(res[j]['name'])
    edges = rSubset(cast_mem)
    for k in range(len(edges)):
        MG.add_edge(edges[k][0], edges[k][1])


# some basic network properties
print('no of nodes: ', len(MG.nodes()))
print('no of edges: ', len(MG.edges()))
conn_comp = list(nx.connected_components(MG))
print('no of connected components:', len(conn_comp))
degree_sequence = sorted([len(n) for n in conn_comp], reverse=True)
print(degree_sequence)
print('The highly connected componenet has no of nodes : ', degree_sequence[0])

# degree histogram
degree_sequence = sorted([d for n, d in MG.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.figure(figsize=(20, 10))
plt.plot(deg, cnt)
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.xlim(0, 300)

plt.show()
visualize(MG)