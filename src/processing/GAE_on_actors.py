"""
Using the graph autoencoder LoNGAE on our actors network. Goals:
1. Predicting new high-score relations between actors
2. Suggesting alternative actors using their latent values in the GAE
3. Rating a group of actors
"""

from src.modelling.LoNGAE.train_lp_with_feats import run
from src.processing.actors_network_features import get_actors_network_features
from src.utils.TM_dataset import actors_feature_balancing_weight
from datetime import datetime
import numpy as np
import paths
from tensorflow import keras
from src.modelling.LoNGAE.models.ae import autoencoder_with_node_features


def train():
    time_zero = datetime.now()
    actors_adjacency, actors_feature, actors_id = get_actors_network_features()
    node_features_weight = actors_feature_balancing_weight()
    print('delta T: ', datetime.now() - time_zero)
    time_zero = datetime.now()

    run(actors_adjacency, actors_feature, node_features_weight, evaluate_lp=True)
    print('delta T: ', datetime.now() - time_zero)


def load_encoder_model(file_path=paths.models+'actors_graph_ae/encoder.keras'):
    return keras.models.load_model(file_path)


def load_autoencoder_model(adj, feats, node_features_weight,
                           file_path=paths.models + 'actors_graph_ae/autoencoder_weights.h5'):
    _, ae = autoencoder_with_node_features(adj.shape[1], feats.shape[1], node_features_weight)
    ae.load_weights(file_path)
    return ae


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


def get_rating_predictor():
    """

    create a function mapping from actor to predicted ratings for its edges and genres
    Note: the model must be trained beforehand
    :return: the rating predictor of actors, and the list of actor ids
    """
    actors_adjacency, actors_feature, actors_id = get_actors_network_features()
    node_features_weight = actors_feature_balancing_weight()
    try:
        ae = load_autoencoder_model(actors_adjacency, actors_feature, node_features_weight)
    except OSError:
        raise Exception('the model must be trained beforehand using the train function')

    def rating_predictor(actor_id):
        """

        :param actor_id:
        :return: two 1d numpy arrays;
        first array is predicted average ratings for its edges with length of number of the nodes in the graph
        second array is predicted average ratings for the actor in each genres
        """
        try:
            actor_idx = actors_id.index(actor_id)
        except ValueError:
            raise Exception('actor id is not found in the graph')
        adj = actors_adjacency[actor_idx]
        feat = actors_feature[actor_idx]
        adj_aug = np.concatenate((adj, feat))
        adj_aug_outs = ae.predict(adj_aug.reshape(1, -1))  # a list with length 2 [decoded_adj, decoded_feats]
        adj_outs, feat_outs = adj_aug_outs[0], adj_aug_outs[1]
        adj_out, feat_out = adj_outs[0], feat_outs[0]  # prediction on one sample
        return adj_out, feat_out

    return rating_predictor, actors_id


def target_actor_weight(target_actor_id, actors_id, predicted_weights):
    try:
        target_idx = actors_id.index(target_actor_id)
    except:
        raise Exception('Actor id {} is not found in the graph'.format(target_actor_id))

    return predicted_weights[target_idx]


if __name__ == '__main__':
    # train()
    # exit()

    rating_predictor, actors_id = get_rating_predictor()
    edges_weights, genres_weights = rating_predictor(actors_id[0])
    print(edges_weights[0])
    print(target_actor_weight(35, actors_id, edges_weights))
    print(genres_weights)
    exit()

    latent_vector_generator, actors_id = get_latent_vector_generator()
    print(latent_vector_generator(actors_id[0]))
