"""
Using the graph autoencoder LoNGAE on our actors network. Goals:
1. Predicting new high-score relations between actors
2. Suggesting alternative actors using their latent values in the GAE
3. Rating a group of actors
"""

from src.modelling.actors_network import get_network
from src.utils.TM_dataset import actor_rating_genre_based
import numpy as np
from src.modelling.LoNGAE.train_lp_with_feats import run


if __name__ == '__main__':
    actors_adjacency, actors_id = get_network()
    actors_feature = [actor_rating_genre_based(a_id) for a_id in actors_id]
    actors_feature = np.array(actors_feature)
    run(actors_adjacency, actors_feature)
