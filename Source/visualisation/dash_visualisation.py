import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from visualisation_utils import model_search, prepare_data

import pandas as pd


def create_plots(augmented_data):
    """
    Create the plots for visualisation

    :param augmented_data: df,
        the data augmented with statistics
    :return: plotly figures
    """
    # Line plot of true and predicted labels
    data_0 = augmented_data[augmented_data['variable'].isin(['True_labels', 'Response'])]
    fig_0 = px.line(data_0, x="Cycle", y="value", color='variable', title='True vs predicted labels')

    # Line plot of run times
    data_1 = augmented_data[augmented_data['variable'].isin(['Run_times'])]
    fig_1 = px.line(data_1, x="Cycle", y="value", color='variable', title='Run times')

    # Line plot of accuracy
    data_2 = augmented_data[augmented_data['variable'].isin(['Predicted_CMA', 'Response_CMA'])]
    fig_2 = px.line(data_2, x="Cycle", y="value", color='variable', title='Cumulative Moving Average')

    # Pie chart of result types
    data_3 = augmented_data[augmented_data['variable'].isin(['Prediction_result'])]
    fig_3 = px.pie(data_3, values='Cycle', names='value', title='Prediction results')

    # Pie chart of response types
    data_4 = augmented_data[augmented_data['variable'].isin(['Response_result'])]
    fig_4 = px.pie(data_4, values='Cycle', names='value', title='System response results')

    return fig_0, fig_1, fig_2, fig_3, fig_4


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    # header
    html.Div([
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url("dash-new-logo.png"),
                        className="app__menu__img",
                    )
                ],
                className="app__header__logo",
            ),
        ],
        className="app__header",
    ),
    html.Div([
        html.Div([
            html.H3(children='Search for the best model with given window size'),
            dcc.Input(id="model_window", type="number", placeholder="Model window size", value=500),
            html.Button('Find model', id='model_button')
        ]),
        html.Div([
            html.H3(children='Search for the best model with given window size'),
        ], id='Chosen_model'),
    ]),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='simulation_data', style={'display': 'none'}),

    html.Div([
        html.H3(children='Sliding window size'),
        dcc.Slider(
                id='window_slider',
                min=0,
                max=2001,
                value=500,
                marks={str(x): str(x) for x in range(0, 2001, 50)},
                step=None
        )
    ]),

    html.Div([
        html.H3(children='Sliding window type'),
        dcc.Dropdown(
                id='window_dropdown',
                options=[

                ],
                value='hamming'
        )
    ]),

    html.Div([
        # html.H1(children='Hello Dash'),
        #
        # html.Div(children='''
        #     Dash: A web application framework for Python.
        # '''),
        dcc.Graph(
                id='fig_0', config={"displayModeBar": False}
        ),
    ], id='container_0'),
    html.Div([
        dcc.Graph(
                id='fig_1'
        ),
    ], id='container_1'),
    html.Div([
        dcc.Graph(
                id='fig_2'
        ),
    ], id='container_2'),
    html.Div([
        html.Div([
            dcc.Graph(
                    id='fig_3'
            ),
        ], className='six columns', id='container_3'),
        html.Div([
            dcc.Graph(
                    id='fig_4'
            ),
        ], className='six columns', id='container_4'),
    ])
], id='figures')


@app.callback([Output('simulation_data', 'children'),
               Output('chosen_model', 'children')],
              [Input('model_button', 'n_clicks'),
               Input('model_window', 'value')])
def load_data(clicks, value):
    # To determine if n_clicks is changed.
    changed_ids = [p['prop_id'].split('.')[0] for p in dash.callback_context.triggered]
    button_pressed = 'model_button' in changed_ids

    if not button_pressed:
        return "", ""

    if value == "":  # Do nothing if button is clicked and input num is blank.
        return "No input", ""

    df, chosen_model = model_search(value)

    if isinstance(df, pd.DataFrame):
        return df.to_json(orient='split'), chosen_model
    else:
        return dash.no_update, chosen_model


@app.callback(
        [Output('fig_0', 'figure'),
         Output('fig_1', 'figure'),
         Output('fig_2', 'figure'),
         Output('fig_3', 'figure'),
         Output('fig_4', 'figure')],
        [Input('simulation_data', 'children'),
         Input('window_slider', 'value'),
         Input('window_dropdown', 'value')])
def update_figure(simulation_data, window_size, window_type):
    if simulation_data == '':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    data = pd.read_json(simulation_data, orient='split')

    filtered_df = prepare_data(data, window_size, window_type)

    fig_0, fig_1, fig_2, fig_3, fig_4 = create_plots(filtered_df)

    return fig_0, fig_1, fig_2, fig_3, fig_4


if __name__ == '__main__':
    app.run_server(debug=True)
    app.title = "Smart Industry simulation analysis"
