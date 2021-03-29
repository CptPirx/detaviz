__doc__ = """
Lightweight TABL model for anomaly detection
"""

import sys

sys.path.append('../')

from abstract_classifier import Abstract_Classifier
from TABL import Layers

from tensorflow import keras as k
import tensorflow as tf


class TABLModel(Abstract_Classifier):
    def __init__(self, window, dimensions, classes, optimizer, dropout, projection_regularizer, projection_constraint,
                 attention_regularizer, attention_constraint, n_bl_layers, bl_layers, n_tabl_layers, tabl_layers):
        self.window = window
        self.dimensions = dimensions
        self.classes = classes
        self.optimizer = optimizer
        self.dropout = dropout
        self.projection_regularizer = projection_regularizer
        self.projection_constraint = projection_constraint
        self.attention_regularizer = attention_regularizer
        self.attention_constraint = attention_constraint
        self.n_bl_layers = n_bl_layers
        self.bl_layers = bl_layers
        self.n_tabl_layers = n_tabl_layers
        self.tabl_layers = tabl_layers

        self.model = self.construct_model()

    def construct_model(self):
        inputs_class = k.layers.Input(shape=(self.window, self.dimensions))
        x = Layers.BL((self.window, self.dimensions),
                      self.projection_regularizer,
                      self.projection_constraint)(inputs_class)
        x = k.layers.Activation('relu')(x)
        x = k.layers.Dropout(self.dropout)(x)

        x = inputs_class
        for i in range(0, self.n_bl_layers):
            x = Layers.BL(self.bl_layers[i],
                          self.projection_regularizer,
                          self.projection_constraint)(x)
            x = k.layers.Activation('relu')(x)
            x = k.layers.Dropout(self.dropout)(x)

        for i in range(0, self.n_tabl_layers):
            x = Layers.TABL(self.tabl_layers[i],
                            self.projection_regularizer,
                            self.projection_constraint,
                            self.attention_regularizer,
                            self.attention_constraint)(x)

        outputs_class = k.layers.Activation('softmax')(x)

        model = k.Model(inputs=inputs_class, outputs=outputs_class)
        # model.compile(optimizer=self.optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
        model.compile(optimizer=self.optimizer, loss=tf.keras.losses.BinaryCrossentropy())
        model.summary()

        return model

    def get_config(self):
        config = super().get_config()
        config.update({
            'projection_regularizer': self.params['projection_regularizer'],
            'projection_constraint': self.params['projection_constraint'],
            'attention_regularizer': self.params['attention_regularizer'],
            'attention_constraint': self.params['attention_constraint']
        })

        return config
