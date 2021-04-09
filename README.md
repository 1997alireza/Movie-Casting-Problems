# Movie-Casting-Problems

An attempt to use movie-related datasets to solve movie casting problems for movie producers or directors. 
The implemented models in this project can help them to find and choose an appropriate cast for their movie.

The main questions we have answered in this project are:
- If a choosen actor is not available for a movie, who is the best alternative actor? [`Alternative Actor Suggestion`](src/processing/alternative_actor_suggestion)

- Which actors perfectly match each other to play the main roles of a movie? [`Co-Star Suggestion`](src/processing/co_star_suggestion.py)

- How to compare two casts to be choosen for a movie? [`Movie Cast Rating`](src/processing/movie_cast_rating.py)

To answer these questions, we have created a graph named Actors Network and applied the graph autoencoder LoNGAE to this network.
Mean squared error of our model on the test data for link weight prediction is 0.005406, after 50 epochs for weights between 0 and 1.
<br>

## Actors Network
Edge weights and node features of actors network are created using the movie ratings extracted from provided datasets.
<p align="center">
<img src="docs/images/actors-network.png?raw=True" alt="Actors Network" width="70%"/>
</p>

##  Local Neighborhood Graph Autoencoder (LoNGAE)
LoNGAE is available in the reposetory [graph-representation-learning](https://github.com/vuptran/graph-representation-learning).
Here you can see how this autoencoder is applied on our graph.
<p align="center">
<img src="docs/images/LoNGAE.png?raw=True" alt="LoNGAE"/>
</p>
