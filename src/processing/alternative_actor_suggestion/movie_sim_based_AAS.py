from src.modelling.movie_similarity import get_index_from_title, MG
import networkx as nx


def get_alternative_actors(actor):
    if get_index_from_title(actor) is None:
        candids = nx.descendants_at_distance(MG, actor, 3)
        result = ""
        for candid in candids:
            if get_index_from_title(candid) is None:
                result = result + ", " + candid
        print(actor + "  ?  " + result)
        return [candid]
    return []
