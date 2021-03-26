import abc
import numpy as np
import pickle

import evaluator

import tensorflow.keras as k


class Abstract_Classifier:
    """
    The abstract classifier class. Unifies the training procedure

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def construct_model(self):
        """

        :return:
        """
        pass

    def train(self, train_generator, epochs):
        """
        Train the model

        :param model:
        :param train_generator:
        :param epochs:
        :return:
        """
        # Tensorboard Callback
        tb_callback = k.callbacks.TensorBoard('./logs', update_freq=1)

        # Early stopping callback
        earlystop_callback = k.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0.0001,
                patience=6
        )

        # Reduce learning rate on plateau callback
        learning_rate_reduction_callback = k.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                         patience=4,
                                                                         verbose=1,
                                                                         factor=0.5,
                                                                         min_lr=0.0001)

        history = self.model.fit(train_generator,
                                 epochs=epochs,
                                 callbacks=[earlystop_callback,
                                            learning_rate_reduction_callback,
                                            tb_callback],
                                 use_multiprocessing=False,
                                 verbose=2,
                                 workers=8)

        self.model.save("model")

    def score(self, test_labels, test_generator, mode):
        """

        :param test_labels:
        :return:
        """
        test_labels = test_labels[self.window:]
        predicted_probabilities = self.model.predict(test_generator)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        metrics = evaluator.lob_evaluator(test_labels, predicted_labels, mode)
        evaluator.save_plots(predicted_labels, predicted_probabilities, test_labels, mode)

        return metrics
