"""
utils to work with the movies database (TMDB)
"""

import pandas as pd
import paths
import numpy as np
from sklearn.preprocessing import normalize

__links_df = pd.read_csv(paths.the_movies_dataset + 'links.csv', usecols=['tmdbId', 'movieId'])
__ratings_df = pd.read_csv(paths.the_movies_dataset + 'ratings.csv', usecols=['movieId', 'rating'])
__credits = pd.read_csv(paths.the_movies_dataset + 'credits.csv', usecols=['id', 'cast'])  # this id is tmdb_id
__movies = pd.read_csv(paths.the_movies_dataset + 'movies_metadata.csv', usecols=['id', 'genres', 'vote_average'])  # this id is tmdb_id


# list of the genres with high number of movies in the movies dataset
__top_genres_list = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary', 'Adventure',
                     'Science Fiction', 'Family', 'Mystery', 'Fantasy', 'Animation', 'Foreign', 'Music', 'History',
                     'War', 'Western', 'TV Movie']


# def rating_of_movie(tmdb_id):
#     """
#     ratings.movieId = links.movieId; so we need to find the related moveId for the input tmdbId using links data
#     :param tmdb_id: it's equal to links.tmdbId or movies_metadata.id, and credits.id as well
#     :return: the average of the ratings. if there is not any rating available for that movie, it returns NaN
#     """
#     global __links_df, __ratings_df
#
#     try:
#         movie_id = __links_df[__links_df.tmdbId == tmdb_id].iloc[0].movieId
#     except IndexError:
#         raise Exception('tmdb id {} is not found'.format(tmdb_id))
#     ratings = __ratings_df[__ratings_df.movieId == movie_id].rating
#     if len(ratings):
#         raise Exception('no rating has been found for the movie {}'.format(tmdb_id))
#
#     return __ratings_df[__ratings_df.movieId == movie_id].rating.mean()


def rating_of_movie(tmdb_id):
    """

    :param tmdb_id: an integer which is equal to links.tmdbId or movies_metadata.id, and credits.id as well
    :return: normalized average vote (movies_metadata.vote_average/10)
    """
    global __movies
    try:
        normalized_rating = __movies[__movies['id'] == str(tmdb_id)].iloc[0]['vote_average'] / 10
        if normalized_rating == 0.0:
            raise Exception('the movie {} is not rated'.format(tmdb_id))
        else:
            return normalized_rating
    except Exception:
        raise Exception('the movie {} is not rated'.format(tmdb_id))


def genres_of_movie(tmdb_id):
    """

    :param tmdb_id:
    :return: list of related genres to the movie
    """
    global __movies
    tmdb_id = str(tmdb_id)

    try:
        genres = eval(__movies[__movies.id == tmdb_id].iloc[0].genres)
    except IndexError:
        raise Exception('tmdb id {} is not found'.format(tmdb_id))

    genres = [g['name'] for g in genres]
    return genres


def in_top_genres(tmdb_id):
    genres = genres_of_movie(tmdb_id)
    for g in genres:
        if g in __top_genres_list:
            return True
    return False


def actor_rating_genre_based(actor_id):
    """

    :param actor_id: it's equal to credits.cast.id
    :return: normalized actor's feature based on its ratings in top genres (__genres_list), as a numpy array
    """
    actor_id = int(actor_id)
    genres = dict((g, 0) for g in __top_genres_list)
    for i in range(len(__credits)):
        casts_id = [c['id'] for c in eval(__credits['cast'][i])]
        if actor_id in casts_id:
            movie_id = __credits['id'][i]
            movie_genres = genres_of_movie(movie_id)

            try:
                movie_rating = rating_of_movie(movie_id)
            except Exception:  # no rating has been found for the movie
                continue

            for g in movie_genres:
                if g in genres:
                    genres[g] += movie_rating

    feature = np.array(list(genres.values()))
    total = sum(feature)

    if total == 0:
        raise Exception('actor id {} is not found in any rated movie\'s casts from top genres.'.format(actor_id))

    return feature / total


def actors_rating_genre_based(actor_ids):
    """
    more efficient than actor_rating_genre_based. here we read the dataset just once.
    :param actor_ids: a list of actor_id which is equal to credits.cast.id
    :return: list of normalized actors' features based on their ratings in top genres (__genres_list), as a 2d numpy array
    """
    features = np.zeros((len(actor_ids), len(__top_genres_list)))

    for i in range(len(__credits)):
        movie_id = __credits['id'][i]
        movie_genres = genres_of_movie(movie_id)
        genres_idx = []
        for g in movie_genres:
            try:
                g_idx = __top_genres_list.index(g)
                genres_idx.append(g_idx)
            except ValueError:
                continue

        if len(genres_idx) == 0:  # the movie is not in any top genre
            continue

        try:
            movie_rating = rating_of_movie(movie_id)
        except Exception:  # no rating has been found for this movie
            continue

        casts_id = [c['id'] for c in eval(__credits['cast'][i])]
        for cast_id in casts_id:
            try:
                a_idx = actor_ids.index(cast_id)
            except ValueError:
                continue

            features[a_idx][genres_idx] += movie_rating

    print('Actors features are extracted.')
    return normalize(features)
