import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import paths


def counting_values(df, column):
    value_count = {}
    for row in df[column].dropna():
        if row in value_count:
            value_count[row] += 1
        else:
            value_count[row] = 1
    return value_count


if __name__ == '__main__':
    movie_meta = pd.read_csv(paths.the_movies_dataset + '/movies_metadata.csv', low_memory=False)
    votes_count = pd.Series(counting_values(movie_meta, 'vote_average'))
    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(votes_count)) % cmap.N)
    fig = plt.figure()
    votes_count.sort_values(ascending=False).plot(kind='bar', color=colors)
    fig.show()
    
