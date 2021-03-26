# Import required libraries
import copy
import dash
import datetime
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from visualisation_utils import model_search, prepare_data

# Multi-dropdown options
from controls import window_options

app = dash.Dash(
        __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Create global chart template
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
        autosize=True,
        automargin=True,
        margin=dict(l=30, r=30, b=20, t=40),
        hovermode="closest",
        plot_bgcolor="#F9F9F9",
        paper_bgcolor="#F9F9F9",
        legend=dict(font=dict(size=10), orientation="h"),
        title="Satellite Overview",
        mapbox=dict(
                accesstoken=mapbox_access_token,
                style="light",
                center=dict(lon=-78.05, lat=42.54),
                zoom=7,
        ),
)

# Create app layout
app.layout = html.Div(
        [
            # Hidden div inside the app that stores the intermediate value
            html.Div(id='simulation_data', style={'display': 'none'}),

            # empty Div to trigger javascript file for graph resizing
            html.Div(id="output-clientside"),
            html.Div(
                    [
                        html.Div(
                                [
                                    html.Img(
                                            src=app.get_asset_url("au_logo.png"),
                                            id="au-image",
                                            style={
                                                "height": "60px",
                                                "width": "auto",
                                                "margin-bottom": "25px",
                                            },
                                    )
                                ],
                                className="one-third column",
                        ),
                        html.Div(
                                [
                                    html.Div(
                                            [
                                                html.H3(
                                                        "Smart Industry",
                                                        style={"margin-bottom": "0px"},
                                                ),
                                                html.H5(
                                                        "Simulation Analysis", style={"margin-top": "0px"}
                                                ),
                                            ]
                                    )
                                ],
                                className="one-third  column",
                                id="title",
                        ),
                        html.Div(
                                [
                                    html.A(
                                            html.Button("GitLab repo", id="learn-more-button"),
                                            href="https://gitlab.au.dk/smart-industry/anomaly_simulation",
                                    )
                                ],
                                className="one-third column",
                                id="button",
                        ),
                    ],
                    id="header",
                    className="row flex-display",
                    style={"margin-bottom": "25px"},
            ),
            html.Div(
                    [
                        html.Div([
                            html.P("Select model window size and dimensionality:", className="control_label"),
                            dcc.Input(id="model_window",
                                      type="number",
                                      placeholder="Model window size",
                                      min=0,
                                      value=500,
                                      className="dcc_control"),
                            # html.P("Select model dimensionality:", className="control_label"),
                            dcc.Input(id="model_dimensionality",
                                      type="number",
                                      placeholder="Model dimensionality",
                                      max=125,
                                      min=0,
                                      value=60,
                                      className="dcc_control"),
                            html.P("Select system response window type:", className="control_label"),
                            dcc.Dropdown(
                                    id='window_type',
                                    options=window_options,
                                    value='hamming',
                                    className="dcc_control",
                            ),
                        ],
                                className="pretty_container four columns",
                                id="cross-filter-options",
                        ),
                        html.Div(
                                [
                                    html.Div(
                                            [
                                                html.Div(
                                                        [html.H6(id="chosen_model_text"),
                                                         html.P("Chosen model")],
                                                        id="chosen_model",
                                                        className="mini_container",
                                                ),
                                                html.Div(
                                                        [html.H6(id="avg_acc_text"),
                                                         html.P("Model accuracy")],
                                                        id="avg_f1",
                                                        className="mini_container",
                                                ),
                                                html.Div(
                                                        [html.H6(id="simulation_avg_acc_text"),
                                                         html.P("Simulation accuracy")],
                                                        id="simulation_avg_acc",
                                                        className="mini_container",
                                                ),
                                                html.Div(
                                                        [html.H6(id="simulation_length_text"),
                                                         html.P("Simulation length")],
                                                        id="simulation_length",
                                                        className="mini_container",
                                                ),
                                            ],
                                            id="info-container",
                                            className="row container-display",
                                    ),
                                    html.Div(
                                            [html.P(
                                                    "Select simulation window size:",
                                                    className="control_label",
                                            ),
                                                dcc.Slider(
                                                        id='window_slider',
                                                        min=0,
                                                        max=2001,
                                                        value=500,
                                                        marks={str(x): str(x) for x in range(0, 2001, 50)},
                                                        step=None,
                                                        className="dcc_control"
                                                ), ],
                                            id="windowSizeContainer",
                                            className="pretty_container",
                                    ),
                                ],
                                id="right-column",
                                className="eight columns",
                        ),
                    ],
                    className="row flex-display",
            ),
            html.Div(
                    [
                        html.Div(
                                [dcc.Graph(id="results_graph")],
                                className="pretty_container seven columns",
                        ),
                        html.Div(
                                [dcc.Graph(id="avg_accuracy_graph")],
                                className="pretty_container five columns",
                        ),
                    ],
                    className="row flex-display",
            ),
            html.Div(
                    [
                        html.Div(
                                [dcc.Graph(id="pie_graph")],
                                className="pretty_container seven columns",
                        ),
                        html.Div(
                                [dcc.Graph(id="run_times_graph")],
                                className="pretty_container five columns",
                        ),
                    ],
                    className="row flex-display",
            ),
        ],
        id="mainContainer",
        style={"display": "flex", "flex-direction": "column"},
)


# Helper functions
def produce_statistics(augmented_data):
    simulation_length = str(datetime.timedelta(minutes=augmented_data.shape[0] / 60000))
    simulation_avg_acc = np.round(augmented_data['value'].loc[augmented_data['variable'] == 'Response_accuracy'].mean(),
                                  decimals=3)

    return simulation_avg_acc, simulation_length


def create_plots(augmented_data):
    """
    Create the plots for visualisation

    :param augmented_data: df,
        the data augmented with statistics
    :return: plotly figures
    """
    # Line plot of true and predicted labels
    data_0 = augmented_data[augmented_data['variable'].isin(['True_labels', 'Response'])]
    layout_labels = copy.deepcopy(layout)

    data = [
        dict(
                type="scatter",
                mode="lines",
                name="True labels",
                x=data_0['Cycle'],
                y=data_0['value'].loc[data_0['variable'] == 'True_labels'],
                line=dict(shape="spline", smoothing="2", color="#F9ADA0"),
        ),
        dict(
                type="scatter",
                mode="lines",
                name="System response",
                x=data_0['Cycle'],
                y=data_0['value'].loc[data_0['variable'] == 'Response'],
                line=dict(shape="spline", smoothing="2", color="#849E68"),
        ),
    ]
    layout_labels["title"] = "System response vs true labels"

    fig_0 = dict(data=data, layout=layout_labels)

    # Line plot of run times
    data_1 = augmented_data[augmented_data['variable'].isin(['Run_times'])]
    layout_runtimes = copy.deepcopy(layout)

    data = [
        dict(
                type="scatter",
                mode="lines",
                name="Prediction CMA",
                x=data_1['Cycle'],
                y=data_1['value'].loc[data_1['variable'] == 'Run_times'],
                line=dict(shape="spline", smoothing="2", color="#849E68"),
        ),
    ]
    layout_runtimes["title"] = "Simulation run times"

    fig_1 = dict(data=data, layout=layout_runtimes)

    # Line plot of accuracy
    data_2 = augmented_data[augmented_data['variable'].isin(['Predicted_CMA', 'Response_CMA'])]
    layout_accuracy = copy.deepcopy(layout)

    data = [
        dict(
                type="scatter",
                mode="lines",
                name="Prediction CMA",
                x=data_2['Cycle'],
                y=data_2['value'].loc[data_2['variable'] == 'Predicted_CMA'],
                line=dict(shape="spline", smoothing="2", color="#F9ADA0"),
        ),
        dict(
                type="scatter",
                mode="lines",
                name="System CMA",
                x=data_2['Cycle'],
                y=data_2['value'].loc[data_2['variable'] == 'Response_CMA'],
                line=dict(shape="spline", smoothing="2", color="#849E68"),
        ),
    ]
    layout_accuracy["title"] = "Prediction vs system Cumulative Moving Average accuracy"

    fig_2 = dict(data=data, layout=layout_accuracy)

    # Pie chart of result types
    data_3 = augmented_data[augmented_data['variable'].isin(['Prediction_result'])]
    data_3 = data_3['value'].value_counts()
    data_4 = augmented_data[augmented_data['variable'].isin(['Response_result'])]
    data_4 = data_4['value'].value_counts()

    layout_pie = copy.deepcopy(layout)

    data = [
        dict(
                type="pie",
                labels=["False positive", "False negative", "Correct"],
                values=[data_3['False positive'],
                        data_3['False negative'],
                        data_3['Correct']],
                name="Simulation response",
                text=[
                    "False positives",
                    "False negatives",
                    "Correct",
                ],
                hoverinfo="text+value+percent",
                textinfo="label+percent+name",
                hole=0.5,
                marker=dict(colors=["#92d8d8", "#fac1b7", "#a9bb95"]),
                domain={"x": [0, 0.45], "y": [0.2, 0.8]},
        ),
        dict(
                type="pie",
                labels=["False positive", "False negative", "Correct"],
                values=[data_4['False positive'],
                        data_4['False negative'],
                        data_4['Correct']],
                name="System response",
                text=[
                    "False positives",
                    "False negatives",
                    "Correct",
                ],
                hoverinfo="text+value+percent",
                textinfo="label+percent+name",
                hole=0.5,
                marker=dict(colors=["#92d8d8", "#fac1b7", "#a9bb95"]),
                domain={"x": [0.55, 1], "y": [0.2, 0.8]},
        ),
    ]
    layout_pie["title"] = "Results breakdown: simulation vs system response"
    layout_pie["font"] = dict(color="#777777")
    layout_pie["legend"] = dict(
            font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
    )

    fig_3 = dict(data=data, layout=layout_pie)

    return fig_0, fig_1, fig_2, fig_3


@app.callback([Output('simulation_data', 'children'),
               Output('chosen_model_text', 'children'),
               Output('avg_acc_text', 'children')],
              [Input('model_window', 'value'),
               Input('model_dimensionality', 'value')])
def load_data(model_window, model_dimensionality):
    if model_window == "":  # Do nothing if button is clicked and input num is blank.
        return "", "No input", 0

    df, chosen_model, max_acc = model_search(model_window, model_dimensionality)

    if isinstance(df, pd.DataFrame):
        return df.to_json(orient='split'), chosen_model, max_acc
    else:
        return dash.no_update, chosen_model, max_acc


@app.callback(
        [Output('results_graph', 'figure'),
         Output('run_times_graph', 'figure'),
         Output('avg_accuracy_graph', 'figure'),
         Output('pie_graph', 'figure'),
         Output("simulation_avg_acc_text", "children"),
         Output("simulation_length_text", "children"),
         ],
        [Input('simulation_data', 'children'),
         Input('window_slider', 'value'),
         Input('window_type', 'value')])
def update_figure(simulation_data, window_size, window_type):
    if simulation_data == '':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, 0, 0

    data = pd.read_json(simulation_data, orient='split')

    augmented_df = prepare_data(data, window_size, window_type)

    fig_0, fig_1, fig_2, fig_3 = create_plots(augmented_df)

    simulation_avg_acc_text, simulation_length_text = produce_statistics(augmented_df)

    return fig_0, fig_1, fig_2, fig_3, simulation_avg_acc_text, simulation_length_text


# Main
if __name__ == "__main__":
    # app.run_server(debug=True)
    app.run_server(debug=True)
