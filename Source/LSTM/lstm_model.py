__doc__ = """
Lightweight LSTM model for anomaly detection
"""

import meta
import math
import time
import os

from Data_loaders.data_loader import get_dataset_tf

import numpy as np
import pandas as pd

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras import layers, Sequential, losses
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from tensorflow import keras


# Use mixed precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


class LSTMModel(object):
    def __init__(self, type, window, horizon, lstm_size, model_path, train_x, train_y, test_x, test_y, dataset_labels):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.dataset_labels = dataset_labels
        self.type = type
        self.horizon = horizon
        self.window = window
        self.lstm_size = lstm_size
        self.model_path = model_path

    def create_model_stateful(self):
        """
        Create a stateful LSTM model

        :return:
        """
        self.model = Sequential()
        self.model.add(layers.LSTM(self.lstm_size, batch_input_shape=(1, self.window, self.train_x.shape[1]), stateful=True))
        self.model.add(layers.Dense(self.horizon))
        self.model.compile(loss=losses.CategoricalCrossentropy(), optimizer='adam')
        self.model.summary()

    def create_model_regular(self):
        """
        Create a regular LSTM model

        :return:
        """
        self.model = Sequential()
        self.model.add(layers.LSTM(self.lstm_size, input_shape=(self.window, self.train_x.shape[2])))
        self.model.add(layers.Dense(self.train_y.shape[1]))
        self.model.compile(loss=losses.CategoricalCrossentropy(), optimizer='adam')
        self.model.summary()

    def train_stateful(self):
        """
        Train the stateful model

        :return:
        """
        for i in range(meta.epochs):
            self.model.fit(self.train_x, self.train_y, epochs=1, batch_size=1, shuffle=False)
            self.model.reset_states()

        self.model.save("model_stateful.h5")

    def train_regular(self):
        """
        Train the regular model

        :return:
        """
        # Early stopping callback
        earlystop_callback = keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.0001,
            patience=6
        )

        # Reduce learning rate on plateau callback
        learning_rate_reduction_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                             patience=4,
                                                                             verbose=1,
                                                                             factor=0.5,
                                                                             min_lr=0.0001)

        start = time.time()
        history = self.model.fit(self.train_x, self.train_y,
                                 epochs=meta.epochs,
                                 batch_size=meta.batch_size,
                                 callbacks=[earlystop_callback,
                                            learning_rate_reduction_callback],
                                 use_multiprocessing=True,
                                 workers=3)
        end = time.time()

        print('Training done in {time} s'.format(time=(end - start)))

        # Save the model
        self.model.save(self.model_path+'/model.h5')

        # Add training time to history
        history.history['time'] = end - start
        # Save the model's history
        pd.DataFrame.from_dict(history.history).to_csv(self.model_path+'/history.csv', index=False)

    def forecast_lstm(self, x, n_batch):
        """
        Make a single forecast

        :param x: data
        :param n_batch: int, batch size
        :return: array of forecasts
        """
        # reshape input pattern to [samples, timesteps, features]
        x = x.reshape(1, len(x), 1)
        # make forecast
        forecast = self.model.predict(x, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    def make_forecasts(self, n_batch):
        """


        :param n_batch: int, batch size
        :return:
        """
        forecasts = list()
        for i in range(self.test_x.shape[0]):
            # make forecast
            forecast = self.forecast_lstm(self.test_x[i, :], n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    def evaluate_forecasts(self, forecasts):
        """
        Evaluate the accuracy for each forecast time step

        :param forecasts: array of forecasts
        :return:
        """
        for i in range(self.horizon):
            actual = [row[i] for row in self.dataset_labels]
            predicted = [forecast[i] for forecast in forecasts]
            if actual == predicted:
                score = True
            else:
                score = False
            print('t+%d score: %f ' % ((i + 1), score))

    def return_model(self):
        return self.model

    def plot_single_horizon(self, forecasts):
        """
        Plot results of forecasting with horizon size 1

        :param forecasts: data
        :return:
        """
        plt.plot(self.dataset_labels[:meta.plot_length], label='Original')
        plt.plot(forecasts[:meta.plot_length], label='Forecasts')
        plt.legend(loc="upper left")
        plt.show()

    def plot_multi_horizon(self, forecasts):
        """
        Plot results of forecasting with horizon longer than 1

        :param forecasts:
        :return:
        """
        plot_forecasts = []
        for i in range(0, len(forecasts[:meta.plot_length]), 10):
            plot_forecasts.append(forecasts[i])

        plt.plot(self.dataset_labels[:meta.plot_length], label='Original')
        plt.plot(plot_forecasts, label='Forecasts')
        plt.legend(loc="upper left")
        plt.show()

    def run_model(self):
        if self.type == 'stateful':
            self.create_model_stateful()
            self.train_stateful()
            forecasts = self.make_forecasts(n_batch=1)
            self.evaluate_forecasts(forecasts=forecasts)
            self.plot_multi_horizon(forecasts)
        elif self.type == 'regular' and self.horizon == 1:
            self.create_model_regular()
            self.train_regular()
            forecasts = self.model.predict(self.test_x)
            #self.evaluate_forecasts(forecasts=forecasts)
            self.plot_single_horizon(forecasts)
        else:
            self.create_model_regular()
            self.train_regular()
            # forecasts = self.make_forecasts(n_batch=meta.batch_size)
            # forecasts = self.scaler.inverse_transform(forecasts)
            # self.test_y = self.scaler.inverse_transform(self.test_y)
            # self.evaluate_forecasts(forecasts=forecasts)
            # self.plot_multi_horizon(forecasts)

