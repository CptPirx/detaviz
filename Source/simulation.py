__doc__ = """
Script responsible for running the entire simulation.
"""

# Tensorflow logging level
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import load_model

# Use mixed precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

# Allow memory growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import Zoo.meta as meta

from sensor import Sensor
from hapDev import HapDev

from Zoo.TABL.Layers import BL, TABL
from Source.visualisation import plot_simulation_history

from pathlib import Path

import tqdm
import aursad


def main():
    """

    :return:
    """
    # print(device_lib.list_local_devices())
    # Define simulation parameters
    print('Define the simulation parameters')

    # The cycle count
    cycle_count = int(input('Define number of cycles. Default=1000: ') or 1000)

    # The model type
    model_type = input('Enter the model type. Default=tabl: ') or 'tabl'

    # The model source
    while True:
        model_dir = None
        model_source = input('Enter the first symbols of folder name in Results/runs: ')
        for x in os.listdir('../Zoo/Results/runs'):
            if x.startswith(model_source):
                model_dir = x
        if model_dir is not None:
            break
        else:
            print('No such folder')
            continue

    # The sensor domain
    while True:
        domain = int(input('Which sensor to use, 0 to 4. Default=0: ') or 0)
        if domain > 4:
            print('Wrong sensor id.')
            continue
        else:
            break

    # The model's window size
    window = int(input('Enter the model window size.Default=500: ') or 500)

    # The model's prediction horizon
    horizon = int(input('Enter the horizon size. Default=1: ') or 1)

    # The data's dimensionality
    n_dim = int(input('Enter the data dimensionality. Default=60: ') or 60)

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

    _, _, test_x, test_y = aursad.get_dataset_numpy(path=Path(meta.data_path),
                                                    reduce_dimensionality=True,
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
                                                    onehot_labels=False)

    # Create the device
    device = HapDev(buffer_size=buffer,
                    network_delay=meta.network_delay,
                    window=window,
                    horizon=horizon,
                    threshold=threshold)

    # Create a sensor
    sensor = Sensor(axle_nr=domain, buffer_size=buffer, dataset=test_x, dataset_labels=test_y)
    device.add_sensor(sensor)

    # Load the ML model
    if model_type == 'tabl':
        custom_objects = {'BL': BL,
                          'TABL': TABL,
                          'MaxNorm': tf.keras.constraints.max_norm}
        model = load_model(Path('../Zoo/Results/runs/' + model_dir + '/model'), custom_objects=custom_objects)
    else:
        model = load_model(Path('../Zoo/Results/runs/' + model_dir + '/model'))

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

    results_path = '../Results/' + model_source
    Path(results_path).mkdir(parents=False, exist_ok=True)
    # Plot the results
    plot_simulation_history(sent_samples, plot_labels, errors, run_times, results_path, domain, cycle_count)


if __name__ == '__main__':
    main()
