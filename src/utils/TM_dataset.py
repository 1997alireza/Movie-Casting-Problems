"""
utils to work with the movies database (TMDB)
"""

import pandas as pd
import paths

__links_df = pd.read_csv(paths.the_movies_dataset + 'links.csv', usecols=['tmdbId', 'movieId'])
__ratings_df = pd.read_csv(paths.the_movies_dataset + 'ratings.csv', usecols=['movieId', 'rating'])


def rating_of_movie(tmdb_id):
    """
    ratings.movieId = links.movieId; so we need to find the related moveId for the input tmdbId using links data
    :param tmdb_id: based in the dataset it's equal to links.tmdbId or movies_metadata.id, and credits.id as well
    :return: the average of the ratings. if there is not any rating available for that movie, it returns NaN
    """
    global __links_df, __ratings_df

    try:
        movie_id = __links_df[__links_df.tmdbId == tmdb_id].iloc[0].movieId
    except IndexError:
        raise Exception('tmdb id {} is not found'.format(tmdb_id))

    return __ratings_df[__ratings_df.movieId == movie_id].rating.mean()
