import sys

sys.path.append('../')

from abstract_classifier import Abstract_Classifier

from tensorflow import keras as k
import tensorflow as tf


class ResNetModel(Abstract_Classifier):
    def __init__(self, window, dimensions, classes, optimizer, n_feature_maps):
        self.window = window
        self.dimensions = dimensions
        self.classes = classes
        self.optimizer = optimizer
        self.n_feature_maps = n_feature_maps

        self.model = self.construct_model()

    def construct_model(self):
        input_layer = k.layers.Input(shape=(self.window, self.dimensions))

        # BLOCK 1

        conv_x = k.layers.Conv1D(filters=self.n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = k.layers.BatchNormalization()(conv_x)
        conv_x = k.layers.Activation('relu')(conv_x)

        conv_y = k.layers.Conv1D(filters=self.n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = k.layers.BatchNormalization()(conv_y)
        conv_y = k.layers.Activation('relu')(conv_y)

        conv_z = k.layers.Conv1D(filters=self.n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = k.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = k.layers.Conv1D(filters=self.n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = k.layers.BatchNormalization()(shortcut_y)

        output_block_1 = k.layers.add([shortcut_y, conv_z])
        output_block_1 = k.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = k.layers.Conv1D(filters=self.n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = k.layers.BatchNormalization()(conv_x)
        conv_x = k.layers.Activation('relu')(conv_x)

        conv_y = k.layers.Conv1D(filters=self.n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = k.layers.BatchNormalization()(conv_y)
        conv_y = k.layers.Activation('relu')(conv_y)

        conv_z = k.layers.Conv1D(filters=self.n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = k.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = k.layers.Conv1D(filters=self.n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = k.layers.BatchNormalization()(shortcut_y)

        output_block_2 = k.layers.add([shortcut_y, conv_z])
        output_block_2 = k.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = k.layers.Conv1D(filters=self.n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = k.layers.BatchNormalization()(conv_x)
        conv_x = k.layers.Activation('relu')(conv_x)

        conv_y = k.layers.Conv1D(filters=self.n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = k.layers.BatchNormalization()(conv_y)
        conv_y = k.layers.Activation('relu')(conv_y)

        conv_z = k.layers.Conv1D(filters=self.n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = k.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = k.layers.BatchNormalization()(output_block_2)

        output_block_3 = k.layers.add([shortcut_y, conv_z])
        output_block_3 = k.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = k.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = k.layers.Dense(self.classes, activation='softmax')(gap_layer)

        model = k.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=self.optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
        model.summary()

        return model
