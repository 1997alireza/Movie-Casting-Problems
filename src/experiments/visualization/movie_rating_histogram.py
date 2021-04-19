import pandas as pd
import plotly.express as px

import paths

__movie_meta = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', low_memory=False)
d = pd.read_csv(paths.the_movies_dataset + '/cast_rating.csv', low_memory=False)


# d = pd.read_csv('/Users/ho3in/Desktop/' + '100_movie_cast_rating.csv', low_memory=False)

# def counting_values(df, column):
#     """
#     counts given column on the df distribution
#     :return: value_count: shows how many times a value has been repeated
#     """
#     value_count = {}
#     for row in df[column].dropna():
#         if row == 0.0:
#             pass
#         elif row in value_count:
#             value_count[row] += 1
#         else:
#             value_count[row] = 1
#     return pd.DataFrame.from_dict(value_count, orient='index')


def ratings_histogram():
    """
    plots a histogram on movie ratings distribution
    :return:
    """
    global __movie_meta
    fig = px.histogram(__movie_meta[__movie_meta['vote_average'] > 0], x='vote_average',
                       labels={'x': 'ratings', 'y': 'count'})
    fig.show()


def cast_rating_histogram():
    fig = px.histogram(d, x='rating',
                       labels={'x': 'cast_rating', 'y': 'count'})
    fig.show()


if __name__ == '__main__':
    ratings_histogram()
