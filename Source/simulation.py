__doc__ = """
Script responsible for running the entire simulation.
"""

__version__ = """ 
v0.1.0 First release
"""

import os

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import tqdm
import click
import tensorflow as tf
import aursad
import multiprocessing as mp
import logging
import pandas as pd

from tensorflow.keras.models import load_model
from pathlib import Path, PurePath

from sensor import Sensor
from hapDev import HapDev

sys.path.append('../')
import Zoo.meta as meta
from Zoo.TABL.Layers import BL, TABL
from Source.visualisation.visualisation import plot_simulation_history


def validate_model_exists(ctx, param, value):
    if ctx.params['run_mode'] == 'single':
        found = False
        for x in os.listdir('../Zoo/Results/runs'):
            if x.startswith(value):
                ctx.params['model_dir'] = x
                found = True
                break

        if not found:
            raise click.BadParameter('no such model exists')


@click.command()
@click.version_option(version=__version__)
@click.option('--cycle_count', default=50000, type=int, help='Number of simulation cycles.')
@click.option('--binary_labels', default=True, type=bool, help='True for binary labels, False for multi-class')
@click.option('--model_dir',
              default=None,
              callback=validate_model_exists,
              help='The first symbols of folder name in Results/runs')
@click.option('--window', default=500, type=int, help='Rolling window size')
@click.option('--n_dim', default=60, type=int, help='Data dimensionality')
@click.option('--n_cpu', default=mp.cpu_count() - 1, type=int, help='Number of threads to use if running '
                                                                                 'multiple simulations')
@click.option('--run_mode', default='single', type=str, help='Determines how many simulations will be run: single will '
                                                             'run 1 simulation with a specified model, all will run '
                                                             'all models, fill will run all models that do not have'
                                                             'simulation results yet.')
@click.option('--verbose', default=0, type=int, help='Logging level, from 0 (debug) to 3 (error)')
def main(cycle_count, binary_labels, model_dir, window, n_dim, n_cpu, run_mode):
    """
    Starts the processes and runs the simulations

    :param cycle_count:
    :param binary_labels:
    :param model_dir:
    :param window:
    :param n_dim:
    :param n_cpu:
    :param run_mode:
    :return:
    """
    single_simulation(cycle_count=cycle_count,
                      binary_labels=binary_labels,
                      model_dir=model_dir,
                      window=window,
                      n_dim=n_dim,
                      n_cpu=n_cpu,
                      run_mode=run_mode)


def setup_logging():
    pass


def single_simulation(cycle_count, binary_labels, model_dir, window, n_dim, n_cpu, run_mode):
    """
    Run a single system simulation

    :param cycle_count:
    :param binary_labels:
    :param model_dir:
    :param window:
    :param n_dim:
    :param n_cpu:
    :param run_mode:
    :return:
    """

    # The sensor domain
    # while True:
    #     domain = int(input('Which sensor to use, 0 to 4. Default=0: ') or 0)
    #     if domain > 4:
    #         print('Wrong sensor id.')
    #         continue
    #     else:
    #         break
    domain = 0

    # The model's prediction horizon
    # horizon = int(input('Enter the horizon size. Default=1: ') or 1)
    horizon = 1

    # The model's buffer size
    # while True:
    #     buffer = int(input('Enter the buffer size. Buffer >= window: '))
    #     if buffer < window:
    #         print('Buffer needs to be at least as big as window.')
    #         continue
    #     else:
    #         break
    buffer = window

    # Whether to reduce dimensionality
    if n_dim < 125:
        reduce_dim = True
    else:
        reduce_dim = False

    _, _, test_x, test_y = aursad.get_dataset_numpy(path=Path(meta.data_path),
                                                    reduce_dimensionality=reduce_dim,
                                                    n_dimensions=n_dim,
                                                    subsample_data=True,
                                                    subsample_freq=2,
                                                    pad_data=False,
                                                    normal_samples=1,
                                                    damaged_samples=1,
                                                    assembly_samples=1,
                                                    missing_samples=1,
                                                    damaged_thread_samples=0,
                                                    loosening_samples=0,
                                                    drop_extra_columns=True,
                                                    onehot_labels=False,
                                                    binary_labels=binary_labels)

    # Create the device
    device = HapDev(buffer_size=buffer,
                    network_delay=meta.network_delay,
                    window=window,
                    horizon=horizon)

    # Create a sensor
    sensor = Sensor(domain=domain,
                    buffer_size=buffer,
                    dataset=test_x,
                    dataset_labels=test_y,
                    label_counter=window)
    device.add_sensor(sensor)

    # Load the ML model
    custom_objects = {'BL': BL,
                      'TABL': TABL,
                      'MaxNorm': tf.keras.constraints.max_norm}
    model = load_model(Path('../Zoo/Results/runs/' + model_dir + '/model'), custom_objects=custom_objects)
    model = load_model(Path('../Zoo/Results/runs/' + model_dir + '/model'))

    device.receive_model(model)

    # Lists to hold simulation results
    accuracy_list = []
    predicted_labels = []
    true_labels = []
    run_times = []
    debug_times = []

    # Run x steps of simulation
    for i in tqdm.tqdm(range(cycle_count), desc='Running simulation cycles'):
        if sensor.check_end():
            print('Reached the end of the dataset!')
            break
        # print("Cycle number {i}".format(i=i))
        accuracy, predicted_label, true_label, run_time, debug_results = device.run_one_cycle(domain)
        debug_times.append(debug_results)
        accuracy_list.append(accuracy)
        predicted_labels.append(predicted_label)
        true_labels.append(true_label)
        run_times.append(run_time)

    model_name = PurePath(model_dir)
    results_path = '../Results/' + model_name.name
    Path(results_path).mkdir(parents=False, exist_ok=True)

    # Save the simulation data
    simulation_results = {'True_labels': true_labels,
                          'Predicted_labels': predicted_labels,
                          'Accuracy': accuracy_list,
                          'Run_times': run_times}

    # Save debug data
    debug_df = pd.DataFrame(debug_times)
    debug_df.to_csv(results_path + '/debug_times.csv')

    simulation_results_df = pd.DataFrame(simulation_results)
    simulation_results_df.to_csv((results_path + '/Results_cycles-{cycle_count}_sensorID-{sensor}.csv').format(
            cycle_count=cycle_count,
            sensor=domain
    ), index_label='Cycle')

    # Plot the results
    plot_simulation_history(predicted_labels, true_labels, accuracy_list, run_times, results_path, domain, cycle_count)


if __name__ == '__main__':
    main()
