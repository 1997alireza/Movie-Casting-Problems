from src.processing.GAE_on_actors import get_rating_predictor


def suggest_co_stars(actor_id, top_num=4):
    """

    :param actor_id:
    :param top_num: size of the returned list
    :return: the id of the best co-stars for actor_id with their co-acting rating
    """
    rating_predictor, actors_id = get_rating_predictor()
    if actor_id not in actors_id:
        raise Exception('actor id {} is not found in the graph'.format(actor_id))

    adj_out, _ = rating_predictor(actor_id)

    adj_pair = zip(range(len(adj_out)), adj_out)
    sorted_adj = sorted(adj_pair, key=lambda i: i[1], reverse=True)[:top_num]
    return [(actors_id[a_id], w) for a_id, w in sorted_adj]
