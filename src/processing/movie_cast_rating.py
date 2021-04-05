import pandas as pd
import paths

from src.modelling.actors_network import parse_movie_cast, get_all_pairs
from src.processing.GAE_on_actors import get_rating_predictor, target_actor_weight
from src.utils.TM_dataset import genres_of_movie, get_genre_index

__movies = pd.read_csv(paths.the_movies_dataset + 'movies_metadata.csv',
                       usecols=['id', 'original_title'])
__credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv', usecols=['id', 'cast'])
__rating_predictor, __actors_id = get_rating_predictor()
__movies_names = __movies_ids = {}


def is_in_graph(actor):
    try:
        __actors_id.index(actor)
        return True
    except:
        return False


def predict(actor_a, actor_b):
    """returns predict of actor_a -> actor_b in our model"""
    edges_weights, __ = __rating_predictor(actor_a)
    return target_actor_weight(actor_b, __actors_id, edges_weights)


def rating(actor, genre):
    """returns rating of the actor in the given genre in our model"""
    global __rating_predictor
    __, genres_weights = __rating_predictor(actor)
    return genres_weights[get_genre_index(genre)]


def ws(actor_a, actor_b, genres):
    """actor pair genre score"""
    ws = 0
    for genre in genres:
        ws += rating(actor_a, genre) + rating(actor_b, genre)
    return ws


def s(actor_a, actor_b):
    """actor pair score"""
    return (predict(actor_a, actor_b) + predict(actor_b, actor_a)) / 2


def get_movie_cast_rating(movie_id, actor_depth):
    """
    compute a movie cast rating based on its genres and actors relation predictions
    :param movie_id
    :param actor_depth
    :return: cast_score
    """
    try:
        cast = __credits[__credits['id'] == movie_id]['cast'].values.tolist()[0]
        cast = filter(is_in_graph, parse_movie_cast(cast, actor_depth))
        movie_genres = genres_of_movie(movie_id)
        return cast_group_rating(cast, movie_genres)
    except:
        raise Exception('no scoring available')


def cast_group_rating(cast, movie_genres):
    """
    compute a movie cast-score based on given cast and the movie genres
    :param cast: list of actor_ids
    :param movie_genres: the genres of the movie
    :return: cast_score
    """
    try:
        cast_pairs = get_all_pairs(cast)
        total_sws_score = total_ws_score = 0
        for actor_a, actor_b in cast_pairs:
            ws_score = ws(actor_a, actor_b, movie_genres)
            total_sws_score += (s(actor_a, actor_b) * ws_score)
            total_ws_score += ws_score

        # weighted average of actors pairs scores with their genre scores as weights
        return total_sws_score / total_ws_score

    except Exception:
        raise Exception('no scoring available')


def movie_name(movie_id):
    try:
        return __movies_names[movie_id]
    except:
        raise Exception('movie id not found')


def movie_id(movie_name):
    try:
        return __movies_ids[movie_name]
    except:
        raise Exception('movie name not found')


def prepare_names():
    global __movies_ids, __movies_names
    for i in range(len(__movies)):
        try:
            __movies_ids[__movies['original_title'][i]] = int(__movies['id'][i])
            __movies_names[str(__movies['id'][i])] = __movies['original_title'][i]
        except:
            pass


prepare_names()
