import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd
import numpy as np


def prepare_data():
    data = pd.read_csv('E:/source/repos/anomaly_simulation/Results/27760/Results_cycles-100000_sensorID-0.csv')

    return data


def start_plot(augmented_data):
    """
    Start the server and create plots

    :param augmented_data: df, the original simulation data augmented with various observations
    :return:
    """
    app = dash.Dash(__name__)

    all_labels = augmented_data.True_labels.unique()

    app.layout = html.Div([
        dcc.Checklist(
                id="checklist",
                options=[{"label": x, "value": x}
                         for x in all_labels],
                value=all_labels[:],
                labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id="line-chart"),
    ])

    @app.callback(
            Output("line-chart", "figure"),
            [Input("checklist", "value")])
    def update_line_chart(labels):
        mask = augmented_data.True_labels.isin(labels)
        fig = px.line(augmented_data[mask], x="Cycle", y="True_labels")
        return fig

    app.run_server(debug=True)


if __name__ == '__main__':
    augmented_data = prepare_data()
    start_plot(augmented_data)
