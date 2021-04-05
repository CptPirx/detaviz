import logging
import pandas as pd
import os

from pathlib import Path

from visualisation.visualisation_utils import get_file_list, check_flag_value


def group_models(model_list):
    """
    Groups the models that use identical dataset into separate lists.

    :param model_list:
    :return:
    """
    grouped_models = {}
    for item in model_list:
        grouped_models.setdefault((item['window'], item['binary_labels'], item['n_dim'], item['screwdriver_only']), []).append(item)

    return grouped_models


def list_simulations(fill_mode=True, cycle_count=50000):
    """
    Creates a list of models that will be simulated.

    :param fill_mode: bool,
        If True then models without an existing simulation will be returned, otherwise all models are returned
    :param cycle_count: int,
        number of simulation cycles
    :return: list of dictionaries,
        paths to models which will be simulated along with the information required for the simulation
    """
    model_path = Path(os.path.join(Path(__file__).parents[1], 'Zoo\\Results\\runs\\'))
    results_path = Path(os.path.join(Path(__file__).parents[1], 'Results\\'))

    # Get all files in the runs
    file_list = get_file_list(model_path)

    # Get the run flags
    flags_list = [f for f in file_list if 'flags' in f and 'user_' not in f]
    selected_models = []

    # Read each flag file and create a model dictionary
    for f in flags_list:
        window_size_flag = check_flag_value(f, 'window')
        dimensionality_flag = check_flag_value(f, 'dimensionality')
        binarize_flag = check_flag_value(f, 'binarize')
        if binarize_flag is None:
            binarize_flag = 0
        screwdriver_only_flag = check_flag_value(f, 'screwdriver_only')
        if screwdriver_only_flag is None:
            screwdriver_only_flag = 0

        model_path = os.path.abspath(f)
        split_dir = model_path.split(os.sep)
        s = os.sep
        model_path = s.join(split_dir[:-3])
        name = split_dir[-4]

        model_dict = {'name': name,
                      'model_dir': model_path,
                      'window': window_size_flag,
                      'n_dim': dimensionality_flag,
                      'binary_labels': binarize_flag,
                      'screwdriver_only': screwdriver_only_flag}

        selected_models.append(model_dict)

    if fill_mode:
        # Check for existing simulations
        results_list = get_file_list(results_path)
        results_list = [f for f in results_list if '.csv' in f]
        cycles_target = 'cycles-' + str(cycle_count)
        found_results = []
        for f in results_list:
            f_split = f.split('_')
            if cycles_target in f_split:
                found_results.append(f)

        # Take those models which do not have an existing simulation with required number of cycles
        filtered_models = []
        for model in selected_models:
            exists = False
            if not any(model['name'] in s for s in found_results):
                exists = True
            if exists:
                filtered_models.append(model)

        return filtered_models
    else:
        return selected_models


def setup_logging():
    pass
