import tensorflow as tf
import tensorflow.keras as k


class Params:
    """
    Class containing all required parameters for its classifier
    """

    def __init__(self):
        self.a = [[60, 10], [40, 1], [4, 1]]
        self.b = [[120, 5], [60, 2], [4, 1]]
        self.c = [[240, 5], [120, 1], [4, 1]]
        self.d = [[240, 5], [120, 2], [4, 1]]
        self.e = [[240, 5], [120, 2], [60, 1], [4, 1]]

        self.param_tabl_1 = {'template': self.a,
                             'dropout': 0.1,
                             'projection_regularizer': None,
                             'projection_constraint': k.constraints.max_norm(3.0, axis=0),
                             'attention_regularizer': None,
                             'attention_constraint': k.constraints.max_norm(5.0, axis=1),
                             'batch_size': 512,
                             'classifier_epochs': 100,
                             'learning_rate': 0.001,
                             'optimizer': tf.keras.optimizers.Adam(),
                             'loss_function': tf.keras.losses.CategoricalCrossentropy()}

        self.params_list = [self.param_tabl_1]
