import abc
import os
import tensorflow.keras as k
import numpy as np

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.metrics import precision_recall_fscore_support
from tqdm.keras import TqdmCallback

# Use mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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

        self.model.fit(train_generator,
                       epochs=epochs,
                       callbacks=[earlystop_callback,
                                  learning_rate_reduction_callback],
                       use_multiprocessing=False,
                       verbose=2,
                       workers=8)

    def score(self, test_labels, test_generator):
        """

        :param test_labels:
        :return:
        """
        test_labels = test_labels[100:]

        if len(test_labels.shape) > 1:
            test_labels = np.argmax(test_labels, axis=1)

        predicted_labels = self.model.predict(test_generator)
        predicted_labels = np.argmax(predicted_labels, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(test_labels,
                                                                   predicted_labels, average=None)
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                                               average='macro')

        metrics = {'accuracy': np.sum(test_labels == predicted_labels) / len(test_labels), 'precision': precision,
                   'recall': recall, 'f1': f1, 'precision_avg': precision_avg, 'recall_avg': recall_avg,
                   'f1_avg': f1_avg}

        return metrics
