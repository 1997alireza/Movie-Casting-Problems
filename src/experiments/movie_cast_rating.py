
from src.processing.movie_cast_rating import get_movie_cast_rating, movie_id, movie_name
import paths
import pandas as pd
import csv
import random


def top_cast_ratings():
    top_cast_movies = ['The Big Lebowski', 'The Godfather', '12 Angry Men', 'The Departed', 'The Return of The King',
                       'The Dark Knight', 'Black Hawk Dawn', 'Inception', 'Pulp Fiction', 'American Hustle']

    ratings = {}
    for movie_name in top_cast_movies:
        try:
            rating = get_movie_cast_rating(movie_id(movie_name), 5)
            ratings[movie_name] = rating
        except:
            pass

    # result:
    # The Big Lebowski: 0.6488287370278465
    # The Godfather: 0.632729155711032
    # 12 Angry Men: 0.6251690814605261
    # The Departed: 0.6478975342809675
    # Inception: 0.6398410374304397
    # Pulp Fiction: 0.6077448804333317
    # American Hustle: 0.6945717414391934

    return ratings


def random_movies_cast_ratings():
    from src.processing.movie_cast_rating import __credits
    file_path = paths.logs + '1000_movie_cast_ratings.csv'

    ratings_file = open(file_path, mode='w')
    writer = csv.writer(ratings_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    movie_ids = list(__credits['id'])
    random.shuffle(movie_ids)
    movie_ids = movie_ids[:1100]
    for movie_id in movie_ids:
        try:
            rating = get_movie_cast_rating(movie_id, 5)
            writer.writerow([movie_name(str(movie_id)), str(rating)])
        except Exception:
            pass

    ratings_file.close()


def descending_random_generated_ratings():
    file_path = paths.logs + '1000_movie_cast_ratings.csv'
    df = pd.read_csv(file_path, usecols=['name', 'cast_rating'])
    sorted_ratings = []
    for i in range(len(df)):
        sorted_ratings.append(df['cast_rating'][i])
    return sorted_ratings[::-1]  # convert ascending list tobe descending


def top_casts_percentile():
    """showing the percentile of several high quality casts between a hge amount of random casts"""
    top_ratings = top_cast_ratings()
    sorted_random_ratings = descending_random_generated_ratings()
    for movie_name in top_ratings:
        top_r = top_ratings[movie_name]
        for idx, r in enumerate([*sorted_random_ratings, float('inf')]):
            if r > top_r:
                percentile = int(idx / len(sorted_random_ratings) * 100)
                print('{}: {}, percentile {}%'.format(movie_name, top_r, percentile))
                break

    # result:
    # The Big Lebowski: 0.647287028451108, percentile 93%
    # The Godfather: 0.6583803229268587, percentile 96%
    # 12 Angry Men: 0.597728030674651, percentile 55%
    # The Departed: 0.6616886311115655, percentile 96%
    # Inception: 0.6664521220990737, percentile 97%
    # Pulp Fiction: 0.5992585656312174, percentile 56%
    # American Hustle: 0.6967548573250509, percentile 99%
