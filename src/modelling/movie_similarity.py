import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from netwulf import visualize
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import random

import paths

df_size = 40

movies = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', dtype='unicode')
movies = pd.DataFrame(movies).set_index('id')[0:df_size].copy(deep=True)
credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv')
keywords = pd.read_csv(paths.the_movies_dataset + '/keywords.csv', dtype='unicode')
id_to_row = {}
row_to_id = {}
cosine_sim = []
MG = nx.Graph()
all_cast = []


def calculate_ids():
    i = 0
    for index, row in movies.iterrows():
        id_to_row[int(index)] = i
        row_to_id[i] = int(index)
        i += 1


def combined_features(row):
    combined = row['adult'] + " " + row['original_language']
    genres = ast.literal_eval(row['genres'])
    for genre in genres:
        combined = combined + " " + genre['name']
    # words = ast.literal_eval(
    #     keywords[keywords.id == str(row_to_id[get_index_from_title(row['original_title'])])]['keywords'][
    #         get_index_from_title(row['original_title'])])
    # for word in words:
    #     combined = combined + " " + word['name']
    return combined


def calc_cosine_sim():
    movies["combined_features"] = movies.apply(combined_features, axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(movies["combined_features"])
    return cosine_similarity(count_matrix)


def get_index_from_title(title):
    try:
        return id_to_row[int(movies[movies.title == title].index.values[0])]
    except:
        return None


def get_title_from_index(index):
    return movies[movies.index == str(row_to_id[index])]["title"].values[0]


def get_similar(movie_title="Jumanji"):
    movie_index = get_index_from_title(movie_title)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    # i = 0
    # for movie in sorted_similar_movies:
    #     print(get_title_from_index(movie[0]))
    #     i = i + 1
    #     if i > df_size/2:
    #         break
    return get_title_from_index(sorted_similar_movies[1][0])


def prepare_graph():
    for i in range(df_size):
        movie = get_title_from_index(i)
        MG.add_edge(movie, get_similar(movie))
        print(movie + "  ->  " + get_similar(movie))
        cast = credits[credits.id == int(row_to_id[i])]['cast'][i]
        res = ast.literal_eval(cast)
        cast_mem = []
        for j in range(len(res)):
            if (j > 2):
                break
            cast_mem.append(res[j]['name'])
        all_cast.extend(cast_mem)
        for k in range(len(cast_mem)):
            MG.add_edge(cast_mem[k], movie)


calculate_ids()
cosine_sim = calc_cosine_sim()
prepare_graph()

for actor in all_cast:
    if get_index_from_title(actor) == None:
        candids = nx.descendants_at_distance(MG, actor, 3)
        result = ""
        for candid in candids:
            if get_index_from_title(candid) == None:
                result = result + ", " + candid
        print(actor + "  ?  " + result)

visualize(MG)
