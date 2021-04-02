import pandas as pd
import paths

from src.modelling.actors_network import parse_movie_cast, get_all_pairs
from src.processing.GAE_on_actors import get_rating_predictor, target_actor_weight
from src.utils.TM_dataset import genres_of_movie, get_genre_index

__credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv', usecols=['id', 'cast'])
__rating_predictor, __actors_id = get_rating_predictor()


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
    ws = 0
    for genre in genres:
        ws += rating(actor_a, genre) + rating(actor_b, genre)
    return ws


def s(actor_a, actor_b, genres):
    return (predict(actor_a, actor_b) + predict(actor_b, actor_a)) / 2 * ws(actor_a, actor_b, genres)


def get_cast_rating(movie_id, actor_depth):
    """
    assess a movie cast based on its genres and actors relation predictions
    :param movie_id
    :return: actor_depth: depth used for recognizing important actors
    """
    cast = __credits[__credits['id'] == movie_id]['cast'].values.tolist()[0]
    cast_pairs = get_all_pairs(filter(is_in_graph, parse_movie_cast(cast, actor_depth)))
    genres = genres_of_movie(movie_id)
    s_score = ws_score = 0
    for actor_a, actor_b in cast_pairs:
        s_score += s(actor_a, actor_b, genres)
        ws_score += ws(actor_a, actor_b, genres)
    return s_score / ws_score


if __name__ == '__main__':
    for movie_id in __credits['id']:
        print(get_cast_rating(movie_id, 5))
