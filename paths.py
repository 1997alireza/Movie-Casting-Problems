import os

root = os.path.dirname(__file__).replace('\\', '/')
if root[-1] != '/':
    root = root + '/'

# directories
src = root + 'src/'
models = root + 'models/'
logs = root + 'docs/logs/'
the_movies_dataset = root + 'datasets/the-movies-dataset/'
imdb_movies_extensive_dataset = root + 'datasets/IMDb-movies-extensive-dataset/'
temporal_data = root + 'temporal_data/'

# files
