import numpy as np
import pickle
from sklearn.neighbors._kd_tree import KDTree

import paths
from src.processing.GAE_on_actors import get_latent_vector_generator
from src.utils.TM_dataset import actor_name

__tree = KDTree


def prepare_data():
    global __tree
    try:
        __tree = pickle.load(open(paths.models + 'alternative_actors/actors_kdtree.pkl', "rb"))
    except (OSError, IOError) as e:
        sample_size = 1000
        latent_vector_generator, actors_id = get_latent_vector_generator()
        actors_id = actors_id[0:sample_size]
        vectors = np.array([latent_vector_generator(actor_id) for actor_id in actors_id])
        __tree = KDTree(vectors, leaf_size=2)
        pickle.dump(__tree, open(paths.models + 'alternative_actors/actors_kdtree.pkl', "wb"))


def find_alternates(actor_id, k=3):
    """
    calculates k nearest actor to the requested actor in latent space, currently uses euclidean distance
    :param actor_id: credits.cast.id
    :return: alternative_actors: credits.cast.id of the k most similar actors
    """
    global __tree
    latent_vector_generator, actors_id = get_latent_vector_generator()
    distance, index = __tree.query(np.array([latent_vector_generator(actor_id)]), k)
    # print("index"+str(index))
    print("distance: " + str(distance))
    print("names: " + str([actor_name(actors_id[int(ind)]) for ind in index[0]]))


if __name__ == '__main__':
    prepare_data()
    find_alternates(5, 3)
