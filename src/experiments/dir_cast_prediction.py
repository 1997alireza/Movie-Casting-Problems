import pickle
import time
import paths
import pandas as pd
from src.modelling.dir_cast_bigraph import build_graph, predict_links


def graph_building(dataset_portion):
    start_time = time.time()

    credits_df = pd.read_csv(paths.the_movies_dataset + 'credits.csv', usecols=['id', 'cast', 'crew'])
    credits_df = credits_df.sample(frac=dataset_portion)

    sampled_rows = []
    for row_id, row in credits_df.iterrows():
        sampled_rows.append(row_id)
    with open(paths.temporal_data + 'dir_cast_prediction_eval/credits_sampled_rows.pkl', 'wb') as file:
        pickle.dump(sampled_rows, file)

    dir_cast_graph, director_set, cast_set, director_name, cast_name = build_graph(credits_df)

    print('--bi-graph has been built in {} seconds'.format(time.time() - start_time))
    return dir_cast_graph, director_set, cast_set, director_name, cast_name


def link_prediction(dir_cast_graph, director_set, cast_set):
    start_time = time.time()

    director_cast_potentialities = \
        predict_links(dir_cast_graph, director_set, cast_set,
                      potentiality_threshold=0.000001952545879  # = mean + sd/2 of potentialities
                      # more accurate range for potentialities: (0.000001953283555 ,max=0.000001952578576)
                      )

    with open(paths.temporal_data + 'dir_cast_prediction_eval/link_predictions.pkl', 'wb') as file:
        pickle.dump(director_cast_potentialities, file)

    print('--links are predicted in {} seconds'.format(time.time() - start_time))


def extract_top_predictions(number):
    with open(paths.temporal_data + 'dir_cast_prediction_eval/link_predictions.pkl', 'rb') as file:
        director_cast_potentialities = pickle.load(file)
    top_predicted_edges = sorted(director_cast_potentialities, key=lambda l: l[2])[:number]

    with open(paths.temporal_data + 'dir_cast_prediction_eval/top{}_link_predictions.pkl'.format(number), 'wb') as file:
        pickle.dump(top_predicted_edges, file)


def get_rest_of_credits_df():
    credits_df = pd.read_csv(paths.the_movies_dataset + 'credits.csv', usecols=['id', 'cast', 'crew'])

    with open(paths.temporal_data + 'dir_cast_prediction_eval/credits_sampled_rows.pkl', 'rb') as file:
        sampled_rows = pickle.load(file)

    return credits_df.drop(sampled_rows)


def get_high_rated_edges(credits_df):  # with weight (average rating) equal or bigger than 4.0
    start_time = time.time()
    dir_cast_graph, _, _, _, _ = build_graph(credits_df)
    print('--bi-graph has been built in {} seconds'.format(time.time() - start_time))
    edges = dir_cast_graph.edges.data("weight", default=0)
    return [edge for edge in edges if edge[2] >= 4.0]


def save_high_rated_edges():
    rest_of_credits_df = get_rest_of_credits_df()  # without the rows sampled before
    high_rated_edges = get_high_rated_edges(rest_of_credits_df)
    with open(paths.temporal_data + 'dir_cast_prediction_eval/high_rated_edges_from_rest_credits.pkl', 'wb') as file:
        pickle.dump(high_rated_edges, file)


def evaluate_predicted_edges(number_of_top_predictions):
    with open(paths.temporal_data +
              'dir_cast_prediction_eval/top{}_link_predictions.pkl'.format(number_of_top_predictions), 'rb') as file:
        top_predicted_edges = pickle.load(file)

    with open(paths.temporal_data + 'dir_cast_prediction_eval/high_rated_edges_from_rest_credits.pkl', 'rb') as file:
        high_rated_edges = pickle.load(file)

    corrects_count = 0

    for edge in high_rated_edges:
        director_id = edge[0]
        cast_id = edge[1]

        for predicted_edge in top_predicted_edges:
            if predicted_edge[0] == director_id and predicted_edge[1] == cast_id:
                corrects_count += 1

    print(len(high_rated_edges))
    return corrects_count / len(high_rated_edges)


if __name__ == '__main__':
    # dir_cast_graph, director_set, cast_set, director_name, cast_name = graph_building(dataset_portion=0.25)
    # link_prediction(dir_cast_graph, director_set, cast_set)

    number_of_top_predictions = 10000
    # extract_top_predictions(number_of_top_predictions)  # top 0.1% of predictions

    # save_high_rated_edges()

    print(evaluate_predicted_edges(number_of_top_predictions))


