# Import required libraries
import pickle
import copy
import pathlib
import urllib.request
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px

from visualisation_utils import model_search, prepare_data

# Multi-dropdown options
from controls import window_options

app = dash.Dash(
        __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

layout = {}

# Create app layout
app.layout = html.Div(
        [
            dcc.Store(id="simulation_data"),
            dcc.Store(id="augmented_data"),
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
                                className="one-half column",
                                id="title",
                        ),
                        html.Div(
                                [
                                    html.A(
                                            html.Button("Learn More", id="learn-more-button"),
                                            href="https://plot.ly/dash/pricing/",
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
                            html.P("Select model window size:", className="control_label"),
                            dcc.Input(id="model_window",
                                      type="number",
                                      placeholder="Model window size",
                                      value=500,
                                      className="dcc_control"),
                            html.Button('Find model', id='model_button'),
                            html.P("Select window type:", className="control_label"),
                            dcc.Dropdown(
                                    id='window_dropdown',
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
                                                        [html.H6(id="chosen_model_text"), html.P("Chosen model")],
                                                        id="chosen_model",
                                                        className="mini_container",
                                                ),
                                                html.Div(
                                                        [html.H6(id="avg_f1_text"), html.P("Model average F1")],
                                                        id="avg_f1",
                                                        className="mini_container",
                                                ),
                                                html.Div(
                                                        [html.H6(id="simulation_avg_acc_text"),
                                                         html.P("Simulation average accuracy")],
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
                                                    "Select window size:",
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
def human_format(num):
    if num == 0:
        return "0"

    magnitude = int(math.log(num, 1000))
    mantissa = str(int(num / (1000 ** magnitude)))
    return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]


def produce_statistics(augmented_data):
    simulation_length = augmented_data.shape[0] / 1000
    simulation_avg_acc = augmented_data['Response_accuracy'].mean()
    model_avg_f1 = 0

    return model_avg_f1, simulation_avg_acc, simulation_length


# Create callbacks
app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="resize"),
        Output("output-clientside", "children"),
        [Input("count_graph", "figure")],
)


@app.callback(
        [
            Output("avg_f1_text", "children"),
            Output("simulation_avg_acc_text", "children"),
            Output("simulation_length_text", "children"),
        ],
        [
            State('augmented_data', 'data'),
        ],
)
def update_production_text(augmented_data, chosen_model):
    model_avg_f1, simulation_avg_acc, simulation_length = produce_statistics(augmented_data)
    return model_avg_f1, simulation_avg_acc, simulation_length


# Selectors -> main graph
@app.callback(
        Output("results_graph", "figure"),
        [State("augmented_Data", "value")],
)
def make_results_figure(augmented_data):
    # Line plot of true and predicted labels
    data_0 = augmented_data[augmented_data['variable'].isin(['True_labels', 'Response'])]
    figure = px.line(data_0, x="Cycle", y="value", color='variable', title='True vs predicted labels')

    return figure


# Main graph -> individual graph
@app.callback(Output("individual_graph", "figure"), [Input("main_graph", "hoverData")])
def make_individual_figure(main_graph_hover):
    layout_individual = copy.deepcopy(layout)

    if main_graph_hover is None:
        main_graph_hover = {
            "points": [
                {"curveNumber": 4, "pointNumber": 569, "customdata": 31101173130000}
            ]
        }

    chosen = [point["customdata"] for point in main_graph_hover["points"]]
    index, gas, oil, water = produce_individual(chosen[0])

    if index is None:
        annotation = dict(
                text="No data available",
                x=0.5,
                y=0.5,
                align="center",
                showarrow=False,
                xref="paper",
                yref="paper",
        )
        layout_individual["annotations"] = [annotation]
        data = []
    else:
        data = [
            dict(
                    type="scatter",
                    mode="lines+markers",
                    name="Gas Produced (mcf)",
                    x=index,
                    y=gas,
                    line=dict(shape="spline", smoothing=2, width=1, color="#fac1b7"),
                    marker=dict(symbol="diamond-open"),
            ),
            dict(
                    type="scatter",
                    mode="lines+markers",
                    name="Oil Produced (bbl)",
                    x=index,
                    y=oil,
                    line=dict(shape="spline", smoothing=2, width=1, color="#a9bb95"),
                    marker=dict(symbol="diamond-open"),
            ),
            dict(
                    type="scatter",
                    mode="lines+markers",
                    name="Water Produced (bbl)",
                    x=index,
                    y=water,
                    line=dict(shape="spline", smoothing=2, width=1, color="#92d8d8"),
                    marker=dict(symbol="diamond-open"),
            ),
        ]
        layout_individual["title"] = dataset[chosen[0]]["Well_Name"]

    figure = dict(data=data, layout=layout_individual)
    return figure


# Selectors, main graph -> aggregate graph
@app.callback(
        Output("aggregate_graph", "figure"),
        [
            Input("well_statuses", "value"),
            Input("well_types", "value"),
            Input("year_slider", "value"),
            Input("main_graph", "hoverData"),
        ],
)
def make_aggregate_figure(well_statuses, well_types, year_slider, main_graph_hover):
    layout_aggregate = copy.deepcopy(layout)

    if main_graph_hover is None:
        main_graph_hover = {
            "points": [
                {"curveNumber": 4, "pointNumber": 569, "customdata": 31101173130000}
            ]
        }

    chosen = [point["customdata"] for point in main_graph_hover["points"]]
    well_type = dataset[chosen[0]]["Well_Type"]
    dff = filter_dataframe(df, well_statuses, well_types, year_slider)

    selected = dff[dff["Well_Type"] == well_type]["API_WellNo"].values
    index, gas, oil, water = produce_run_times(selected, year_slider)

    data = [
        dict(
                type="scatter",
                mode="lines",
                name="Gas Produced (mcf)",
                x=index,
                y=gas,
                line=dict(shape="spline", smoothing="2", color="#F9ADA0"),
        ),
        dict(
                type="scatter",
                mode="lines",
                name="Oil Produced (bbl)",
                x=index,
                y=oil,
                line=dict(shape="spline", smoothing="2", color="#849E68"),
        ),
        dict(
                type="scatter",
                mode="lines",
                name="Water Produced (bbl)",
                x=index,
                y=water,
                line=dict(shape="spline", smoothing="2", color="#59C3C3"),
        ),
    ]
    layout_aggregate["title"] = "Aggregate: " + WELL_TYPES[well_type]

    figure = dict(data=data, layout=layout_aggregate)
    return figure


# Selectors, main graph -> pie graph
@app.callback(
        Output("pie_graph", "figure"),
        [
            Input("well_statuses", "value"),
            Input("well_types", "value"),
            Input("year_slider", "value"),
        ],
)
def make_pie_figure(well_statuses, well_types, year_slider):
    layout_pie = copy.deepcopy(layout)

    dff = filter_dataframe(df, well_statuses, well_types, year_slider)

    selected = dff["API_WellNo"].values
    index, gas, oil, water = produce_run_times(selected, year_slider)

    aggregate = dff.groupby(["Well_Type"]).count()

    data = [
        dict(
                type="pie",
                labels=["Gas", "Oil", "Water"],
                values=[sum(gas), sum(oil), sum(water)],
                name="Production Breakdown",
                text=[
                    "Total Gas Produced (mcf)",
                    "Total Oil Produced (bbl)",
                    "Total Water Produced (bbl)",
                ],
                hoverinfo="text+value+percent",
                textinfo="label+percent+name",
                hole=0.5,
                marker=dict(colors=["#fac1b7", "#a9bb95", "#92d8d8"]),
                domain={"x": [0, 0.45], "y": [0.2, 0.8]},
        ),
        dict(
                type="pie",
                labels=[WELL_TYPES[i] for i in aggregate.index],
                values=aggregate["API_WellNo"],
                name="Well Type Breakdown",
                hoverinfo="label+text+value+percent",
                textinfo="label+percent+name",
                hole=0.5,
                marker=dict(colors=[WELL_COLORS[i] for i in aggregate.index]),
                domain={"x": [0.55, 1], "y": [0.2, 0.8]},
        ),
    ]
    layout_pie["title"] = "Production Summary: {} to {}".format(
            year_slider[0], year_slider[1]
    )
    layout_pie["font"] = dict(color="#777777")
    layout_pie["legend"] = dict(
            font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
    )

    figure = dict(data=data, layout=layout_pie)
    return figure


# Selectors -> count graph
@app.callback(
        Output("count_graph", "figure"),
        [
            Input("well_statuses", "value"),
            Input("well_types", "value"),
            Input("year_slider", "value"),
        ],
)
def make_count_figure(well_statuses, well_types, year_slider):
    layout_count = copy.deepcopy(layout)

    dff = filter_dataframe(df, well_statuses, well_types, [1960, 2017])
    g = dff[["API_WellNo", "Date_Well_Completed"]]
    g.index = g["Date_Well_Completed"]
    g = g.resample("A").count()

    colors = []
    for i in range(1960, 2018):
        if i >= int(year_slider[0]) and i < int(year_slider[1]):
            colors.append("rgb(123, 199, 255)")
        else:
            colors.append("rgba(123, 199, 255, 0.2)")

    data = [
        dict(
                type="scatter",
                mode="markers",
                x=g.index,
                y=g["API_WellNo"] / 2,
                name="All Wells",
                opacity=0,
                hoverinfo="skip",
        ),
        dict(
                type="bar",
                x=g.index,
                y=g["API_WellNo"],
                name="All Wells",
                marker=dict(color=colors),
        ),
    ]

    layout_count["title"] = "Completed Wells/Year"
    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = False
    layout_count["autosize"] = True

    figure = dict(data=data, layout=layout_count)
    return figure


@app.callback([Output('simulation_data', 'data'),
               Output('chosen_model', 'value')],
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


@app.callback([Output('simulation_data', 'data'),
               Output('chosen_model', 'value')],
              [Input('model_button', 'n_clicks'),
               Input('model_window', 'value')])
def prepare_data(clicks, value):
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


# Main
if __name__ == "__main__":
    app.run_server(debug=True)
