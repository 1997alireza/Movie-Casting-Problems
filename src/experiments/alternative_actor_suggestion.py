from src.modelling.actors_network import parse_movie_cast
from src.processing.alternative_actor_suggestion.movie_sim_based_AAS import get_alternative_actors
from src.processing.GAE_on_actors import get_rating_predictor
from src.processing.alternative_actor_suggestion.GAE_based_AAS import find_alternates
from src.processing.movie_cast_rating import get_movie_cast_rating, movie_id, cast_group_rating
from src.utils.TM_dataset import get_actor_movies, get_cast, genres_of_movie


def compare_alternative_actor_algorithms_using_cast_rating():
    """Here we try to compute alternative actor for each actor using two algorithms

    we provided: using movie similarity and vector space, then we try to compute score
    of cast if th give actor is replaced by the alternative actor an compare values
    Sample Experiments show 2nd algorithm is better"""
    __, actors_id = get_rating_predictor()
    for actor in actors_id:
        for movie in [get_actor_movies(actor)]:
            try:
                movie_genres = genres_of_movie(movie_id)
                print("original score: " + str(get_movie_cast_rating(movie, 5)))

                cast = parse_movie_cast(get_cast(movie), 5)
                cast.remove(int(actor))
                cast.append(get_alternative_actors(actor))
                print("alg 1 score (graph neighbour based): " + str(cast_group_rating(cast, movie_genres)))

                cast = parse_movie_cast(get_cast(movie_id), 5)
                cast.remove(int(actor))
                cast.append(find_alternates(actor, 1))
                print("alg 2 score (neural network based): " + str(cast_group_rating(cast, movie_genres)))
            except Exception as e:
                print(e)


if __name__ == '__main__':
    compare_alternative_actor_algorithms_using_cast_rating()
