import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import paths

new_movies = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', dtype='unicode')
# new_movies = pd.DataFrame(movies).set_index('id')[1:5].copy(deep=True)
keywords = pd.read_csv(paths.the_movies_dataset + '/keywords.csv', dtype='unicode')

# print(new_movies.shape)
# print(keywords.shape)

def combined_features(row):
    combined = row['genres'] + " " + row['adult']
    return combined

def calc_cosine_sim():
    new_movies["combined_features"] = new_movies.apply(combined_features, axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(new_movies["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)
    return cosine_sim


def get_index_from_title(title):
    return int(new_movies[new_movies.title == title].index.values[0])

def get_title_from_index(index):
    return new_movies[new_movies.index == index]["title"].values[0]

def get_similar(movie_title="Jumanji"):
    movie_index = get_index_from_title(movie_title)
    cosine_sim=calc_cosine_sim()
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    i = 0
    for movie in sorted_similar_movies:
        print(get_title_from_index(movie[0]))
        i = i + 1
        if i > 15:
            break

get_similar("Jumanji")
