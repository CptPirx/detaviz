__doc__ = """
Lightweight TABL model for anomaly detection
"""

import meta
import time
import os

import pandas as pd

import Source.TABL.Layers as Layers
import Source.TABL.tabl_meta as tabl_meta

from tensorflow import keras as k
from matplotlib import pyplot as plt
from tensorflow import keras

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Use mixed precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


class TABLModel(object):
    def __init__(self, type, window, horizon, model_path, train_x, train_y, test_x, test_y, dataset_labels):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.dataset_labels = dataset_labels
        self.type = type
        self.horizon = horizon
        self.window = window
        self.model_path = model_path

        params = tabl_meta.Params()
        self.params = params.params_list[0]

    def create_model_regular(self):
        """
        Create a regular TABL model

        :return:
        """
        # Classifier
        inputs_class = k.layers.Input(shape=(self.window, self.train_x.shape[2]))
        x = Layers.BL((self.window, self.train_x.shape[2]),
                      self.params['projection_regularizer'],
                      self.params['projection_constraint'])(inputs_class)
        x = k.layers.Activation('relu')(x)
        x = k.layers.Dropout(self.params['dropout'])(x)

        x = inputs_class
        for i in range(0, len(self.params['template']) - 1):
            x = Layers.BL(self.params['template'][i],
                          self.params['projection_regularizer'],
                          self.params['projection_constraint'])(x)
            x = k.layers.Activation('relu')(x)
            x = k.layers.Dropout(self.params['dropout'])(x)

        x = Layers.TABL(self.params['template'][-1],
                        self.params['projection_regularizer'],
                        self.params['projection_constraint'],
                        self.params['attention_regularizer'],
                        self.params['attention_constraint'])(x)

        outputs_class = k.layers.Activation('softmax')(x)

        self.model = k.Model(inputs=inputs_class, outputs=outputs_class)
        self.model.compile(optimizer=self.params['optimizer'],
                           loss=self.params['loss_function'])

        self.model.summary()

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
        self.model.save(self.model_path + '/model.h5')

        # Add training time to history
        history.history['time'] = end - start
        # Save the model's history
        pd.DataFrame.from_dict(history.history).to_csv(self.model_path + '/history.csv', index=False)

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

    def get_config(self):
        config = super().get_config()
        config.update({
            'projection_regularizer': self.params['projection_regularizer'],
            'projection_constraint': self.params['projection_constraint'],
            'attention_regularizer': self.params['attention_regularizer'],
            'attention_constraint': self.params['attention_constraint']
        })

        return config

    def run_model(self):
        if self.type == 'regular' and self.horizon == 1:
            self.create_model_regular()
            self.train_regular()
            #forecasts = self.model.predict(self.test_x)
            #self.evaluate_forecasts(forecasts=forecasts)
            #self.plot_single_horizon(forecasts)
        else:
            self.create_model_regular()
            self.train_regular()
            # forecasts = self.make_forecasts(n_batch=meta.batch_size)
            # forecasts = self.scaler.inverse_transform(forecasts)
            # self.test_y = self.scaler.inverse_transform(self.test_y)
            # self.evaluate_forecasts(forecasts=forecasts)
            # self.plot_multi_horizon(forecasts)
