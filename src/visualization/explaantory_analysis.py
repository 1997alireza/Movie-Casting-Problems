import pandas as pd
import plotly.express as px
import paths

__movie_meta = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', low_memory=False)


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
    plots a histogram on movie ratings
    :return:
    """
    global __movie_meta
    fig = px.histogram(__movie_meta[__movie_meta['vote_average'] > 0], x='vote_average',
                       labels={'x': 'ratings', 'y': 'count'})
    fig.show()


if __name__ == '__main__':
    ratings_histogram()
