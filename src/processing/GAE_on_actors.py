"""
Using the graph autoencoder LoNGAE on our actors network. Goals:
1. Predicting new high-score relations between actors
2. Suggesting alternative actors using their latent values in the GAE
3. Rating a group of actors
"""

from src.modelling.actors_network import get_network
from src.utils.TM_dataset import actors_rating_genre_based
from src.modelling.LoNGAE.train_lp_with_feats import run


if __name__ == '__main__':
    actors_adjacency, actors_id = get_network()
    actors_feature = actors_rating_genre_based(actors_id)
    run(actors_adjacency, actors_feature, evaluate_lp=True)
