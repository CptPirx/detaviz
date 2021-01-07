__doc__ = """
Data loader for CSV files. There are 5 files for 5 axles. ~40 points per 1 run. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from .utils import split_sequence
import meta


def get_dataset_pandas():
    """
    Get the data in form of 5 separate pandas dataframe, one for each axle.
    Perform all requried transformations.

    :return: 5 pandas dataframes
    """
    axle_1, axle_2, axle_3, axle_4, axle_5 = load_data()

    visualise_datasets(axle_1, axle_2, axle_3, axle_4, axle_5)

    print(axle_1.dtypes)

    return axle_1, axle_2, axle_3, axle_4, axle_5


def get_dataset_numpy(domain):
    """
    Get the data in form of one numpy array with all time series. Shape is Num_series x Samples

    :return:
    """
    axle_1, axle_2, axle_3, axle_4, axle_5 = load_data()

    # Transform to numpy arrays
    axle_1_np = axle_1['Value'].to_numpy(dtype=float)
    axle_2_np = axle_2['Value'].to_numpy(dtype=float)
    axle_3_np = axle_3['Value'].to_numpy(dtype=float)
    axle_4_np = axle_4['Value'].to_numpy(dtype=float)
    axle_5_np = axle_5['Value'].to_numpy(dtype=float)

    # Get only the values column with the lowest number of samples
    axle_1_np = axle_1_np[:34200]
    axle_2_np = axle_2_np[:34200]
    axle_3_np = axle_3_np[:34200]
    axle_4_np = axle_4_np[:34200]
    axle_5_np = axle_5_np[:34200]

    # Stack the arrays into one
    axles_np = np.column_stack((axle_1_np, axle_2_np, axle_3_np, axle_4_np, axle_5_np))
    axles_np = np.transpose(axles_np)

    axles_np = axles_np[domain, :]

    # Scaler is needed to inverse scale the predictions
    scaler = MinMaxScaler(feature_range=(-1, 1))
    axles_np = axles_np.reshape(-1, 1)
    scaler.fit(axles_np[:22800])
    axles_np = scaler.transform(axles_np)
    axles_np = np.ravel(axles_np)

    sensor_range = np.ptp(axles_np)

    return axles_np, scaler, sensor_range


def get_dataset_tf(domain, window, horizon):
    """
    Get the data in form for LSTM training - train and test datasets.
    Perform all required transformations.

    :return: Tensorflow dataset object
    """
    data, scaler, _ = get_dataset_numpy(domain)

    train_x, train_y = split_sequence(data[0:22800], window, horizon)

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

    test_x, test_y = split_sequence(data[22800:], window, horizon)

    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))

    return train_x, train_y, test_x, test_y, data[22800:], scaler


def load_data():
    """
    Load the data into pandas dataframes

    :return: 5 pandas dataframes
    """
    axle_1 = pd.read_csv(meta.data_path + 'Axle_1.csv',
                         sep=',',
                         header=0,
                         names=['Time', 'Precision', 'Value'],
                         dtype={'Time': 'int',
                                'Precision': 'int',
                                'Value': 'float'})
    axle_2 = pd.read_csv(meta.data_path + 'Axle_2.csv',
                         sep=',',
                         header=0,
                         names=['Time', 'Precision', 'Value'],
                         dtype={'Time': 'int',
                                'Precision': 'int',
                                'Value': 'float'})
    axle_3 = pd.read_csv(meta.data_path + 'Axle_3.csv',
                         sep=',',
                         header=0,
                         names=['Time', 'Precision', 'Value'],
                         dtype={'Time': 'int',
                                'Precision': 'int',
                                'Value': 'float'})
    axle_4 = pd.read_csv(meta.data_path + 'Axle_4.csv',
                         sep=',',
                         header=0,
                         names=['Time', 'Precision', 'Value'],
                         dtype={'Time': 'int',
                                'Precision': 'int',
                                'Value': 'float'})
    axle_5 = pd.read_csv(meta.data_path + 'Axle_5.csv',
                         sep=',',
                         header=0,
                         names=['Time', 'Precision', 'Value'],
                         dtype={'Time': 'int',
                                'Precision': 'int',
                                'Value': 'float'})

    # # Concatenate time to full time with ns
    # axle_1['Time'] = axle_1.Time + axle_1.Precision
    # axle_2['Time'] = axle_2.Time + axle_2.Precision
    # axle_3['Time'] = axle_3.Time + axle_3.Precision
    # axle_4['Time'] = axle_4.Time + axle_4.Precision
    # axle_5['Time'] = axle_5.Time + axle_5.Precision

    # Convert Unix time to timestamps
    axle_1['Time'] = pd.to_datetime(axle_1['Time'], unit='ms')
    axle_2['Time'] = pd.to_datetime(axle_2['Time'], unit='ms')
    axle_3['Time'] = pd.to_datetime(axle_3['Time'], unit='ms')
    axle_4['Time'] = pd.to_datetime(axle_4['Time'], unit='ms')
    axle_5['Time'] = pd.to_datetime(axle_5['Time'], unit='ms')

    # Sort columns by date
    axle_1.sort_values(by='Time', ascending=True, inplace=True)
    axle_2.sort_values(by='Time', ascending=True, inplace=True)
    axle_3.sort_values(by='Time', ascending=True, inplace=True)
    axle_4.sort_values(by='Time', ascending=True, inplace=True)
    axle_5.sort_values(by='Time', ascending=True, inplace=True)

    return axle_1, axle_2, axle_3, axle_4, axle_5


def visualise_datasets(axle_1, axle_2, axle_3, axle_4, axle_5):
    """
    Provide simple visualisation of the datasets

    :param axle_1:
    :param axle_2:
    :param axle_3:
    :param axle_4:
    :param axle_5:
    :return: -
    """
    axle_1.set_index('Time')['Value'].plot()
    axle_2.set_index('Time')['Value'].plot()
    axle_3.set_index('Time')['Value'].plot()
    axle_4.set_index('Time')['Value'].plot()
    axle_5.set_index('Time')['Value'].plot()

    plt.show()


if __name__ == '__main__':
    get_dataset_tf()
