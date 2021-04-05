__doc__ = """
Lightweight LSTM model for anomaly detection
"""
import sys
sys.path.append('../')

from abstract_classifier import Abstract_Classifier

import tensorflow as tf
from tensorflow import keras as k


class LSTM_Model(Abstract_Classifier):
    def __init__(self, window, dimensions, n_layers, start_size, classes, optimizer):
        self.window = window
        self.n_layers = n_layers
        self.dimensions = dimensions
        self.classes = classes
        self.optimizer = optimizer

        self.layers_size = [start_size * 1/2 ** i for i in range(n_layers)]
        self.layers_size = [int(i) for i in self.layers_size]
        self.model = self.construct_model()

    def construct_model(self):
        """
        Create a regular LSTM model

        :return:
        """
        inputs = k.layers.Input(shape=(self.window, self.dimensions))
        x = inputs

        for i in range(self.n_layers - 1):
            x = k.layers.LSTM(units=self.layers_size[i], return_sequences=True)(x)
            x = k.layers.BatchNormalization()(x)

        x = k.layers.LSTM(units=self.layers_size[-1])(x)
        x = k.layers.BatchNormalization()(x)

        out = k.layers.Dense(units=self.classes, activation='softmax')(x)

        model = k.Model(inputs=inputs, outputs=out)
        model.compile(optimizer=self.optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
        model.summary()

        return model


