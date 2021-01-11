__doc__ = """
Script responsible for running the entire simulation.
"""

import meta

from sensor import Sensor
from hapDev import HapDev
from LSTM.lstm_model import LSTMModel
from TABL.tabl_model import TABLModel
from TABL.Layers import BL, TABL
from Support.evaluator import Evaluator
from Data_loaders.robot_data_loader import get_dataset_numpy

import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm


def plot_simulation_history(predicted_labels, true_labels, errors, run_times, model_path, domain, cycle_count):
    """

    :param predicted_labels:
    :param errors:
    :param run_times:
    :param model_path:
    :param threshold:
    :param domain:
    :param cycle_count:
    """
    # Calculate mean of sent samples, errors and run times
    run_times[0] = 0
    mean_errors = [np.mean(errors)] * len(errors)
    mean_time = [np.mean(run_times)] * len(run_times)

    # Plot the data
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)

    # Plot the samples
    ax[0].plot(predicted_labels, label='Predicted labels')
    ax[0].plot(true_labels, label='True labels')
    ax[0].legend(loc='upper left')
    ax[0].set(xlabel='Cycle', ylabel='Label', title='True and predicted labels')

    # # Add text box
    # text_0 = 'Mean value={mean} %'.format(mean=np.round(sent_samples_mean[0], decimals=3))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax[0].text(0.7, 0.95, text_0, transform=ax[0].transAxes, fontsize=14,
    #            verticalalignment='top', bbox=props)

    # Plot the errors
    ax[1].plot(errors, label='Accuracy')
    ax[1].plot(mean_errors, label='Mean accuracy', linestyle='--')
    ax[1].legend(loc='upper left')
    ax[1].set(xlabel='Cycle', ylabel='%', title='Accuracy per cycle')

    # Add text box
    text_1 = 'Mean value={mean} %'.format(mean=(np.round(mean_errors[0], decimals=2) * 100))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.7, 0.95, text_1, transform=ax[1].transAxes, fontsize=14,
               verticalalignment='top', bbox=props)

    # Plot the run time
    ax[2].plot(run_times, label='Time')
    ax[2].plot(mean_time, label='Mean time', linestyle='--')
    ax[2].legend(loc='upper left')
    ax[2].set(xlabel='Cycle', ylabel='Ms', title='Time per cycle')

    # Add text box
    text_2 = 'Mean value={mean} ms'.format(mean=np.round(mean_time[0], decimals=3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[2].text(0.7, 0.95, text_2, transform=ax[2].transAxes, fontsize=14,
               verticalalignment='top', bbox=props)

    # Save the figure
    fig.savefig((model_path + '/Results_cycles-{cycle_count}_sensorID-{sensor}.png').format(
            cycle_count=cycle_count,
            sensor=domain
    ),
            bbox_inches='tight')

    plt.show()


def main():
    """

    :return:
    """
    # print(device_lib.list_local_devices())
    # Define simulation parameters
    print('Define the simulation parameters')

    # The cycle count
    cycle_count = int(input('Define number of cycles. Default=1000: ') or 1000)

    # The model source
    while True:
        model_source = input('"train" to train new model, "load" to load pre-trained model. Default=load: ') or 'load'
        if model_source != 'train' and model_source != 'load':
            print('Wrong model source.')
            continue
        else:
            break

    # The sensor domain
    while True:
        domain = int(input('Which sensor to use, 0 to 4. Default=0: ') or 0)
        if domain > 4:
            print('Wrong sensor id.')
            continue
        else:
            break

    # The model's window size
    window = int(input('Enter the model window size.Default=100: ') or 100)

    # The model's prediction horizon
    horizon = int(input('Enter the horizon size. Default=1: ') or 1)

    # The model type
    model_type = input('LSTM or TABL. Default=TABL: ') or 'TABL'

    model_name = model_type
    model_type = 'regular'

    if model_type == 'LSTM':
        # The model's complexity
        lstm_size = int(input('Enter the number of LSTM units. Default=256: ') or 256)
        model_name += f'-{lstm_size}'

        # The model type
        while True:
            model_type = input('"stateful" or "regular". Default=regular: ') or 'regular'
            if model_type != 'stateful' and model_type != 'regular':
                print('Wrong model type.')
                continue
            else:
                break

    # The model's buffer size
    # while True:
    #     buffer = int(input('Enter the buffer size. Buffer >= window: '))
    #     if buffer < window:
    #         print('Buffer needs to be at least as big as window.')
    #         continue
    #     else:
    #         break

    buffer = window

    # The system's threshold
    while True:
        threshold = input('The system threshold. Format -> "0.00". Default=0.05: ').replace(',', '.') or 0.05
        threshold = float(threshold)
        if threshold > 1.0 or threshold < 0.0:
            print('Wrong threshold value.')
            continue
        else:
            break

    # Check if the folder exists. If not, create it
    model_path = meta.results_path + '/model_{model_type}_{model_name}_window-{window}_horizon-{horizon}'.format(
            model_type=model_type,
            window=window,
            horizon=horizon,
            model_name=model_name)
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Load data
    train_x, train_y, test_x, test_y, dataset, dataset_labels = get_dataset_numpy(path=Path(meta.data_path),
                                                                                  sliding_window=True,
                                                                                  window_size=window,
                                                                                  reduce_dimensionality=True,
                                                                                  n_dimensions=60,
                                                                                  subsample_data=True,
                                                                                  subsample_freq=5,
                                                                                  pad_data=False,
                                                                                  normal_samples=1,
                                                                                  damaged_samples=1,
                                                                                  assembly_samples=1,
                                                                                  missing_samples=1,
                                                                                  damaged_thread_samples=0,
                                                                                  loosening_samples=0,
                                                                                  drop_loosen=False,
                                                                                  drop_extra_columns=True,
                                                                                  label_full=False,
                                                                                  start_frac=0.7)

    # Create the device
    device = HapDev(buffer_size=buffer,
                    network_delay=meta.network_delay,
                    window=window,
                    horizon=horizon,
                    threshold=threshold)

    # Create a sensor
    sensor = Sensor(axle_nr=domain, buffer_size=buffer, dataset=dataset, dataset_labels=dataset_labels)
    device.add_sensor(sensor)

    # Load ot train ML model
    if model_source == 'load':
        try:
            custom_objects = {'BL': BL,
                              'TABL': TABL,
                              'MaxNorm': tf.keras.constraints.max_norm}
            model = load_model(model_path + '/model.h5', custom_objects=custom_objects)
            evaluator = Evaluator()
            predicted_labels = model.predict(test_x)
            metrics = evaluator.lob_evaluator(test_labels=test_y, predicted_labels=predicted_labels)
            evaluator.print_metrics(metrics)
        except Exception:
            print('No such model found. Commencing training.')
            if model_type == 'LSTM':
                lstm = LSTMModel(train_x=train_x,
                                 train_y=train_y,
                                 test_x=test_x,
                                 test_y=test_y,
                                 dataset_labels=dataset_labels,
                                 type=model_type,
                                 horizon=horizon,
                                 window=window,
                                 lstm_size=lstm_size,
                                 model_path=model_path)
                lstm.run_model()
                model = lstm.return_model()
            else:
                tabl = TABLModel(train_x=train_x,
                                 train_y=train_y,
                                 test_x=test_x,
                                 test_y=test_y,
                                 dataset_labels=dataset_labels,
                                 type=model_type,
                                 horizon=horizon,
                                 window=window,
                                 model_path=model_path)
                tabl.run_model()
                model = tabl.return_model()
    else:
        if model_type == 'LSTM':
            lstm = LSTMModel(train_x=train_x,
                             train_y=train_y,
                             test_x=test_x,
                             test_y=test_y,
                             dataset_labels=dataset_labels,
                             type=model_type,
                             horizon=horizon,
                             window=window,
                             lstm_size=lstm_size,
                             model_path=model_path)
            lstm.run_model()
            model = lstm.return_model()
        else:
            tabl = TABLModel(train_x=train_x,
                             train_y=train_y,
                             test_x=test_x,
                             test_y=test_y,
                             dataset_labels=dataset_labels,
                             type=model_type,
                             horizon=horizon,
                             window=window,
                             model_path=model_path)
            tabl.run_model()
            model = tabl.return_model()

    device.receive_model(model)

    # Lists to hold simulation results
    errors = []
    sent_samples = []
    run_times = []

    # Run x steps of simulation
    for i in tqdm.tqdm(range(cycle_count), desc='Running simulation cycles'):
        if sensor.check_end():
            print('Reached the end of the dataset!')
            break
        # print("Cycle number {i}".format(i=i))
        error, samples, run_time = device.run_one_cycle(domain, i)
        errors.append(error)
        sent_samples.append(samples)
        run_times.append(run_time)

    plot_labels = dataset_labels[window:(cycle_count + window)]

    # Plot the results
    plot_simulation_history(sent_samples, plot_labels, errors, run_times, model_path, domain, cycle_count)


if __name__ == '__main__':
    main()
