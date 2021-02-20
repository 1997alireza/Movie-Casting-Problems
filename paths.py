import os

root = os.path.dirname(__file__).replace('\\', '/')
if root[-1] != '/':
    root = root + '/'

# directories
src = root + 'src'
the_movies_dataset = root + 'datasets/the-movies-dataset'
imdb_movies_extensive_dataset = root + 'datasets/IMDb-movies-extensive-dataset'

# files
