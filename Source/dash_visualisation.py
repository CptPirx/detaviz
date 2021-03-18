import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np


def prepare_data():

    rolling_window = 1000

    data = pd.read_csv('E:/source/repos/anomaly_simulation/Results/27760/Results_cycles-50000_sensorID-0.csv')

    # Add description of accuracy
    data.loc[data['Predicted_labels'] == data['True_labels'], 'Prediction_result'] = 'Correct'
    data.loc[(data['Predicted_labels'] == 1) & (data['True_labels'] == 0), 'Prediction_result'] = 'False positive'
    data.loc[(data['Predicted_labels'] == 0) & (data['True_labels'] == 1), 'Prediction_result'] = 'False negative'

    # Add system response
    data['Rolling_mean'] = data['Predicted_labels'].rolling(rolling_window, win_type='gaussian').mean(std=3)
    data['Response'] = data['Rolling_mean'].round(0)

    # Add description of system response
    data.loc[data['Response'] == data['True_labels'], 'Response_result'] = 'Correct'
    data.loc[(data['Response'] == 1) & (data['True_labels'] == 0), 'Response_result'] = 'False positive'
    data.loc[(data['Response'] == 0) & (data['True_labels'] == 1), 'Response_result'] = 'False negative'

    # Add system response accuracy
    data['Response_accuracy'] = np.where(0, 1, data['Response'] == data['True_labels'])

    # Add Cumulative Moving Average of accuracy
    data['Predicted_CMA'] = data['Accuracy'].expanding(min_periods=1).mean()
    data['Response_CMA'] = data['Response_accuracy'].expanding(min_periods=1).mean()

    data = data.dropna()
    augmented_data = pd.melt(data, id_vars=['Cycle'])

    return augmented_data


def start_plot(augmented_data):
    """
    Start the server and create plots

    :param augmented_data: df, the original simulation data augmented with various observations
    :return:
    """
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # Line plot of true and predicted labels
    data_0 = augmented_data[augmented_data['variable'].isin(['True_labels', 'Predicted_labels', 'Response'])]
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

    app.layout = html.Div(children=[
        html.Div([
            # html.H1(children='Hello Dash'),
            #
            # html.Div(children='''
            #     Dash: A web application framework for Python.
            # '''),

            dcc.Graph(
                    id='fig_0',
                    figure=fig_0
            ),
        ]),
        html.Div([

            dcc.Graph(
                    id='fig_1',
                    figure=fig_1
            ),
        ]),
        html.Div([

            dcc.Graph(
                    id='fig_2',
                    figure=fig_2
            ),
        ]),
        html.Div([
            html.Div([

                dcc.Graph(
                        id='fig_3',
                        figure=fig_3
                ),
            ], className='six columns'),
            html.Div([

                dcc.Graph(
                        id='fig_4',
                        figure=fig_4
                ),
            ], className='six columns'),
        ])
    ])

    app.run_server(debug=True)


if __name__ == '__main__':
    augmented_data = prepare_data()
    start_plot(augmented_data)
