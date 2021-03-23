"""
Using the graph autoencoder LoNGAE on our actors network. Goals:
1. Predicting new high-score relations between actors
2. Suggesting alternative actors using their latent values in the GAE
3. Rating a group of actors
"""

from src.modelling.LoNGAE.train_lp_with_feats import run
from src.processing.actors_network_features import get_actors_network_features
from datetime import datetime
import numpy as np
import paths
from tensorflow import keras


def train():
    time_zero = datetime.now()
    actors_adjacency, actors_feature, actors_id = get_actors_network_features()
    print('delta T: ', datetime.now() - time_zero)
    time_zero = datetime.now()

    run(actors_adjacency, actors_feature, evaluate_lp=True)
    print('delta T: ', datetime.now() - time_zero)


def load_encoder_model(file_path=paths.models+'actors_graph_ae/encoder.keras'):
    return keras.models.load_model(file_path)


def load_autoencoder_model(file_path=paths.models+'actors_graph_ae/autoencoder.keras'):
    return keras.models.load_model(file_path)


def get_latent_vector_generator():
    """

    create a function mapping from actor to the latent space
    Note: the model must be trained beforehand
    :return: the latent vectors generator of actors, and the list of actor ids
    """
    try:
        encoder = load_encoder_model()
    except OSError:
        raise Exception('the model must be trained beforehand using the train function')

    actors_adjacency, actors_feature, actors_id = get_actors_network_features()

    def latent_vector_generator(actor_id):
        """

        :param actor_id:
        :return: a numpy array in shape (128,)
        """
        try:
            actor_idx = actors_id.index(actor_id)
        except ValueError:
            raise Exception('actor id is not found in the graph')
        adj = actors_adjacency[actor_idx]
        feat = actors_feature[actor_idx]
        adj_aug = np.concatenate((adj, feat))
        return encoder.predict(adj_aug.reshape(1, -1))[0]  # prediction on one sample

    return latent_vector_generator, actors_id


if __name__ == '__main__':
    latent_vector_generator, actors_id = get_latent_vector_generator()
    print(actors_id)
    print(latent_vector_generator(actors_id[0]))

