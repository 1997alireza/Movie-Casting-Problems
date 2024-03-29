import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from src.processing.GAE_on_actors import get_rating_predictor
from src.processing.movie_cast_rating import rating
from src.utils.TM_dataset import __top_genres_list, __top_genres_movies_count, actor_id

__, actors_id = get_rating_predictor()
actor_names = ['Mel Brooks', 'Jack Lemmon', 'Steve Martin', 'Al Pacino', 'Sylvester Stallone', 'Trevor Howard',
               'Tom Hanks', 'Bridget Fonda', 'Tony Curtis']


def get_actor_ratings(actor_id):
    actor_data = []
    for g_id, genre in enumerate(__top_genres_list):
        actor_data.append(rating(actor_id, genre) / __top_genres_movies_count[g_id])
    return actor_data


# def create_df():
#     sample_count = 100
#     data = []
#     for a_idx, actor in enumerate(actors_id):
#         if a_idx == sample_count:
#             break
#         actor_data = [actor_name(actor)]
#         for g_id, genre in enumerate(__top_genres_list):
#             actor_data.append(rating(actor, genre) / __top_genres_movies_count[g_id])
#         data.append(actor_data)
#     cols = ['Name'] + __top_genres_list
#     df = pd.DataFrame(data, columns=cols)
#     return df


def radar_chart():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='Actor-Genre Radar Chart'),
        html.Div(children='''
            You can see how good plays an actor in the given top genres and compare two actors.
        '''),
        html.Br(),
        html.Div([dcc.Dropdown(
            id='input-actor-a',
            options=[{'label': actor_name, 'value': actor_name} for actor_name in actor_names],
            value='Mel Brooks'
        )],
            style={'width': '28%', 'display': 'inline-block'}),
        html.Div([dcc.Dropdown(
            id='input-actor-b',
            options=[{'label': actor_name, 'value': actor_name} for actor_name in actor_names],
            value='Jack Lemmon'
        )],
            style={'width': '28%', 'display': 'inline-block'}),
        dcc.Graph(
            id='radar-chart'
        )
    ])

    @app.callback(
        Output('radar-chart', 'figure'),
        Input('input-actor-a', 'value'),
        Input('input-actor-b', 'value'))
    def update_figure(input_actor_a, input_actor_b):
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=get_actor_ratings(actor_id(input_actor_a)),
            theta=__top_genres_list,
            fill='toself',
            name=input_actor_a,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatterpolar(
            r=get_actor_ratings(actor_id(input_actor_b)),
            theta=__top_genres_list,
            fill='toself',
            name=input_actor_b,
            hoverinfo='skip'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    # range=[0, 1]
                )),
            showlegend=True
        )
        return fig

    app.run_server(debug=True, use_reloader=False)


if __name__ == '__main__':
    radar_chart()
