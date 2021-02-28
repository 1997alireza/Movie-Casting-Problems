import paths
import pandas as pd

if __name__ == '__main__':

    # TODO: remove
    import pickle, numpy as np
    import matplotlib.pyplot as plt
    with open('./modelling/potentialities_10perc.pkl', 'rb') as file:
        director_cast_potentialities = pickle.load(file)
    # print(director_cast_potentialities)
    director_cast_potentialities = np.array(director_cast_potentialities)
    pots = director_cast_potentialities[:, 2].astype(float)
    print(pots)
    print(np.mean(pots))
    np.histogram(pots)

    _ = plt.hist(pots, bins='auto')  # arguments are passed to np.histogram
    plt.show()
    exit()

    the_movies_df = pd.read_csv(paths.the_movies_dataset + 'movies_metadata.csv',
                                usecols=['imdb_id', 'original_title'])

    imdb_movies_df = pd.read_csv(paths.imdb_movies_extensive_dataset + 'IMDb movies.csv',
                                usecols=['imdb_title_id', 'original_title'])

    joined_df = pd.merge(the_movies_df, imdb_movies_df, how='left', left_on=['imdb_id'], right_on=['imdb_title_id'],
                         suffixes=("_the_movies", "_imdb_movies")).drop(columns=['imdb_title_id'])

    print(len(the_movies_df))  # 45466
    print(len(imdb_movies_df))  # 85855
    print(len(joined_df))  # 45466
    # so, all of the movies in the_movies_dataset are included in imdb_movies_extensive_dataset

    for _, row in joined_df.iterrows():
        print(row)
