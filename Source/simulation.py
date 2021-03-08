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
from Source.visualisation import plot_simulation_history

import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

import tqdm


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

    # The data's dimensionality
    n_dim = int(input('Enter the data dimensionality. Default=60: ') or 60)

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
    # train_x, train_y, test_x, test_y, dataset, dataset_labels = get_dataset_numpy(path=Path(meta.data_path),
    #                                                                               sliding_window=True,
    #                                                                               window_size=window,
    #                                                                               reduce_dimensionality=True,
    #                                                                               n_dimensions=60,
    #                                                                               subsample_data=False,
    #                                                                               subsample_freq=2,
    #                                                                               pad_data=False,
    #                                                                               normal_samples=1,
    #                                                                               damaged_samples=1,
    #                                                                               assembly_samples=1,
    #                                                                               missing_samples=1,
    #                                                                               damaged_thread_samples=0,
    #                                                                               loosening_samples=0,
    #                                                                               drop_loosen=False,
    #                                                                               drop_extra_columns=True,
    #                                                                               label_full=False,
    #                                                                               start_frac=0.0)

    dataset, dataset_labels, train_generator, test_generator = get_dataset_numpy(path=Path(meta.data_path),
                                                                                 sliding_window=True,
                                                                                 window_size=window,
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
        except Exception:
            print('No such model found. Commencing training.')
            if model_type == 'LSTM':
                lstm = LSTMModel(train_generator=train_generator,
                                 test_generator=test_generator,
                                 dataset_labels=dataset_labels,
                                 type=model_type,
                                 horizon=horizon,
                                 dimensions=n_dim,
                                 window=window,
                                 lstm_size=lstm_size,
                                 model_path=model_path)
                lstm.run_model()
                model = lstm.return_model()
            else:
                tabl = TABLModel(train_generator=train_generator,
                                 test_generator=test_generator,
                                 dataset_labels=dataset_labels,
                                 type=model_type,
                                 horizon=horizon,
                                 window=window,
                                 dimensions=n_dim,
                                 model_path=model_path)
                tabl.run_model()
                model = tabl.return_model()
    else:
        if model_type == 'LSTM':
            lstm = LSTMModel(train_generator=train_generator,
                             test_generator=test_generator,
                             dataset_labels=dataset_labels,
                             type=model_type,
                             horizon=horizon,
                             window=window,
                             dimensions=n_dim,
                             lstm_size=lstm_size,
                             model_path=model_path)
            lstm.run_model()
            model = lstm.return_model()
        else:
            tabl = TABLModel(train_generator=train_generator,
                             test_generator=test_generator,
                             dataset_labels=dataset_labels,
                             type=model_type,
                             horizon=horizon,
                             window=window,
                             dimensions=n_dim,
                             model_path=model_path)
            tabl.run_model()
            model = tabl.return_model()

    # Evaluate the model
    evaluator = Evaluator()
    predicted_labels = model.predict(test_generator)
    metrics = evaluator.lob_evaluator(test_labels=dataset_labels, predicted_labels=predicted_labels)
    evaluator.print_metrics(metrics)

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
