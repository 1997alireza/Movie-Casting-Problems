import plotly.graph_objects as go
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from src.processing.GAE_on_actors import get_rating_predictor
from src.processing.movie_cast_rating import rating
from src.utils.TM_dataset import actor_name

sample_count = 100
__, actors_id = get_rating_predictor()
genres = ['Drama', 'Romance', 'Comedy', 'Action', 'Thriller', 'Horror', 'Crime', 'Documentary', 'Adventure',
          'Mystery', 'Fantasy', 'Animation', 'Music', 'History', 'War', 'Western', 'TV Movie']
actor_names = [actor_name(actor)[0] for actor in actors_id[0:sample_count]]


def get_actor_ratings(df, actor):
    filtered_df = df[df.Name == actor]
    filtered_df.pop("Name")
    return filtered_df.values.tolist()[0]


def create_df():
    data = []
    i = 0
    for actor in actors_id:
        if i == sample_count:
            break
        i += 1
        actor_data = []
        actor_data.append(actor_name(actor)[0])
        for genre in genres:
            actor_data.append(rating(actor, genre))
        data.append(actor_data)
    cols = ['Name', 'Drama', 'Romance', 'Comedy', 'Action', 'Thriller', 'Horror', 'Crime', 'Documentary', 'Adventure',
            'Mystery', 'Fantasy', 'Animation', 'Music', 'History', 'War', 'Western', 'TV Movie']
    df = pd.DataFrame(data, columns=cols)
    return df


def radar_chart(df):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='Actor-Genre Radar Chart'),
        html.Div(children='''
            You can see how good is a give actor in the given top genres.
        '''),
        html.Br(),
        html.Div(["Actor A: ",
                  dcc.Dropdown(
                      id='input-actor-a',
                      options=[{'label': actor_name, 'value': actor_name} for actor_name in actor_names],
                      value='Jim Carrey'
                  )]),
        html.Div(["Actor B: ",
                  dcc.Dropdown(
                      id='input-actor-b',
                      options=[{'label': actor_name, 'value': actor_name} for actor_name in actor_names],
                      value='Uma Thurman'
                  )]),
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
            r=get_actor_ratings(df, input_actor_a),
            theta=genres,
            fill='toself',
            name=input_actor_a,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatterpolar(
            r=get_actor_ratings(df, input_actor_b),
            theta=genres,
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
    radar_chart(create_df())
