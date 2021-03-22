"""
statistics about the movies dataset

print(len(director_set))  # 17698
print(len(cast_set))  # 205467
print(len(edges))  # 513485
"""

import pandas as pd
import paths


def ratings_stats():
    __credits = pd.read_csv(paths.the_movies_dataset + '/credits.csv', usecols=['id'])
    __movies = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', usecols=['id', 'vote_average'])
    notrated_count = 0
    rated_count = 0
    for i in range(__credits.shape[0]):
        movie_id = __credits['id'][0]
        try:
            rating = __movies[__movies['id'] == str(movie_id)].iloc[0]['vote_average']
            if rating == 0.0:
                notrated_count += 1
            else:
                rated_count += 1
        except Exception:  # no rating has been found for the movie
            notrated_count += 1
            continue
    print("rated: " + str(rated_count))  # 45476
    print("not rated: " + str(notrated_count))  # 0


