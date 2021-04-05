from src.processing.GAE_on_actors import get_rating_predictor, target_actor_weight
from src.utils.TM_dataset import actor_id

"""Ranking The 10 Most Iconic Actor Duos from https://screenrant.com/iconic-actor-duos-ranking/"""
top_duets = [
    ['Seth Rogen', 'James Franco'],
    ['Chris Evans', 'Scarlett Johansson'],
    ['Will Ferrell', 'John C. Reilly'],
    ['Ben Affleck', 'Matt Damon'],
    ['Ben Stiller', 'Owen Wilson'],
    ['Simon Pegg', 'Nick Frost'],
    ['Emma Stone', 'Ryan Gosling'],
    ['Paul Newman', 'Robert Redford'],
    ['Tom Hanks', 'Meg Ryan'],
    ['Robert De Niro', 'Joe Pesci'],
]


def test_known_costars():
    rating_predictor, actors_id = get_rating_predictor()
    for duet in top_duets:
        try:
            duet_0 = actor_id(duet[0])[0]
            duet_1 = actor_id(duet[1])[0]
            edges_weights, __ = rating_predictor(duet_0)
            print(
                str(duet[0]) + "->" + str(duet[1]) + ": " + str(target_actor_weight(duet_1, actors_id, edges_weights)))
            edges_weights, __ = rating_predictor(duet_1)
            print(
                str(duet[1]) + "->" + str(duet[0]) + ": " + str(target_actor_weight(duet_0, actors_id, edges_weights)))
            print()
            
        except Exception as e:
            print(str(duet) + " " + str(e))


if __name__ == '__main__':
    test_known_costars()
