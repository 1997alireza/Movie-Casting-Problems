import paths
import plotly.express as px
import pickle
from src.processing.GAE_on_actors import get_rating_predictor
import numpy as np


def weight_distribution():
    # the distribution of edges' weight of the actors network
    # is located in /docs/plots/actors network weights.png
    graph_file_path = paths.models + 'actors_network/graph_{}_{}.pkl'.format(5, 2)
    graph = pickle.load(open(graph_file_path, 'rb'))
    graph_weights = []
    for _, _, w in graph.edges.data("weight"):
        graph_weights.append(w)

    fig = px.histogram(graph_weights, x='graph weights')
    fig.show()


def predicted_weight_distribution():
    # the distribution of edges' weight of the predicted actors network by LoNGAE
    # is located in /docs/plots/predicted actors network weights.png

    bin_counts = [0] * int(1.0 / 0.005)  # bins for the range [0., 1.] whose lengths is 0.005

    model_rating_predictor, actors_id = get_rating_predictor()
    for i, actor_id in enumerate(actors_id):
        print('{}/{}'.format(i, len(actors_id)))
        adjacency_vector = model_rating_predictor(actor_id)[0]
        for w in adjacency_vector:
            bin_number = int(w / 0.005)
            bin_counts[bin_number] += 1

    fig = px.bar(x=list(np.arange(0., 1., .005)), y=bin_counts, labels={'x': 'predicted graph weights', 'y': 'count'})
    fig.show()
