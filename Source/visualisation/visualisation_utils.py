import os
import json

import numpy as np
import pandas as pd

from pathlib import Path


def check_flag_value(file, flag):
    """
    Check the window flag size

    :param file:
    :param flag:
    :return:
    """
    with open(file) as f:
        datafile = f.readlines()
    for line in datafile:
        if flag in line:
            line_contents = int(line.split(sep=' ')[1])
            break
        else:
            line_contents = None

    return line_contents


def get_file_list(dirName):
    """
    Create a list of file and sub directories

    :param dirName:
    :return:
    """
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_file_list(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def model_search(model_window=500, model_dimensionality=60, cycles=50000):
    """
    Search for the best performing model for the given window size in the model Zoo

    :param model_window: int,
        model window
    :param model_dimensionality: int,
        model dimensionality
    :param cycles: int,
        the length of the simulation to read
    :return: df,
        simulation data
    """
    model_path = os.path.join(Path(__file__).parents[2], 'Zoo/Results/runs/')
    results_path = os.path.join(Path(__file__).parents[2], 'Results/')

    # Get all files in the runs
    file_list = get_file_list(model_path)

    # Get the run flags
    flags_list = [f for f in file_list if 'flags' in f]
    selected_flags = []

    # Read each flags file and select the ones with appropriate window size
    for f in flags_list:
        window_size = check_flag_value(f, 'window')
        dimensionality = check_flag_value(f, 'dimensionality')
        if window_size == model_window and dimensionality == model_dimensionality:
            selected_flags.append(f)

    if len(selected_flags) > 0:
        # Get the selected directories
        selected_dirs = [os.path.abspath(f) for f in selected_flags]
        for i, path in enumerate(selected_dirs):
            split_dir = path.split(os.sep)
            s = os.sep
            selected_dirs[i] = s.join(split_dir[:-3])

        # Read all test_metrics in those directories
        test_metrics = []
        for path in selected_dirs:
            with open(path + '/test_metrics.json') as json_file:
                metrics = json.load(json_file)
            f1 = metrics['f1_avg']
            name = path.split(os.sep)
            name = name[-1:]
            metric_dict = {'f1': f1,
                           'name': name}
            test_metrics.append(metric_dict)

        # Select the file with highest average F1 score and get its directory
        max_f1 = max(test_metrics, key=lambda x: x['f1'])
        load_list = get_file_list(results_path)
        load_list = [f for f in load_list if '.csv' in f]
        load_list = [d for d in load_list if max_f1['name'][0][:4] in d]

        # Select the simulation run with selected number of cycles
        load_dir = [f for f in load_list if ('_cycles-' + str(cycles)) in f]

        if len(load_dir) > 0:
            # Load the results file
            data = pd.read_csv(load_dir[0])
            return data, max_f1['name']
        else:
            return "", 'Simulation not found'
    else:
        return "", 'Model not found'


def prepare_data(data, rolling_window=1000, window_type='hamming', threshold=0.5):
    """
    Prepare the simulation data for visualisation

    :param df: DataFrame,
        the simulation data
    :param rolling_window: int,
        size of the rolling window
    :param window_type: string,
        type of the window -> https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
    :param threshold: float,
        the decision value for classyfing point as anomalous or normal
    :return: df,
        augmented simulation data
    """
    # Add description of accuracy
    data.loc[data['Predicted_labels'] == data['True_labels'], 'Prediction_result'] = 'Correct'
    data.loc[(data['Predicted_labels'] == 1) & (data['True_labels'] == 0), 'Prediction_result'] = 'False positive'
    data.loc[(data['Predicted_labels'] == 0) & (data['True_labels'] == 1), 'Prediction_result'] = 'False negative'

    # Add system response
    if window_type == 'gaussian':
        data['Rolling_mean'] = data['Predicted_labels'].rolling(rolling_window, win_type=window_type).mean(std=3)
    else:
        data['Rolling_mean'] = data['Predicted_labels'].rolling(rolling_window, win_type=window_type).mean()
    data['Response'] = np.where(data['Rolling_mean'] < threshold, 0, 1)

    # Add description of system response
    data.loc[data['Response'] == data['True_labels'], 'Response_result'] = 'Correct'
    data.loc[(data['Response'] == 1) & (data['True_labels'] == 0), 'Response_result'] = 'False positive'
    data.loc[(data['Response'] == 0) & (data['True_labels'] == 1), 'Response_result'] = 'False negative'

    # Add system response accuracy
    data['Response_accuracy'] = np.where(data['Response'] == data['True_labels'], 1, 0)

    # Add Cumulative Moving Average of accuracy
    data['Predicted_CMA'] = data['Accuracy'].expanding(min_periods=1).mean()
    data['Response_CMA'] = data['Response_accuracy'].expanding(min_periods=1).mean()

    data = data.dropna()
    augmented_data = pd.melt(data, id_vars=['Cycle'])

    return augmented_data
