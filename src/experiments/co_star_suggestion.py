from src.processing.GAE_on_actors import get_rating_predictor, target_actor_weight
from src.utils.TM_dataset import actor_id

"""Union of lists: Ranking The 10 Most Iconic Actor Duos from: https://screenrant.com/iconic-actor-duos-ranking/
15 Acting Duos Who Often Film Movies Together, and We’re Happy to See Them Too from:
https://brightside.me/wonder-films/15-acting-duos-who-often-film-movies-together-and-were-happy-to-see-them-too-634760/"""

top_duos = [
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
    ['Martin Freeman', 'Benedict Cumberbatch'],
    ['Anne Hathaway', 'Jake Gyllenhaal'],
    ['Ryan Gosling', 'Emma Stone'],
    ['Steve Carell', 'Christian Bale'],
    ['Bradley Cooper', 'Jennifer Lawrence'],
    ['Johnny Depp', 'Helena Bonham Carter'],
    ['Al Pacino', 'Robert De Niro'],
    ['Matthew McConaughey', 'Woody Harrelson']
]


def test_known_costars():
    rating_predictor, actors_id = get_rating_predictor()
    for duo in top_duos:
        try:
            duo_0 = actor_id(duo[0])
            duo_1 = actor_id(duo[1])
            edges_weights, __ = rating_predictor(duo_0)
            print(
                str(duo[0]) + "->" + str(duo[1]) + ": " + str(target_actor_weight(duo_1, actors_id, edges_weights)))
            edges_weights, __ = rating_predictor(duo_1)
            print(
                str(duo[1]) + "->" + str(duo[0]) + ": " + str(target_actor_weight(duo_0, actors_id, edges_weights)))
            print()

        except Exception as e:
            print(str(duo) + " " + str(e))


if __name__ == '__main__':
    test_known_costars()
