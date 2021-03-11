import sys

sys.path.append('../')

import meta as meta
import tensorflow.keras as k
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from lstm_model import LSTM_Model
import aursad
import numpy as np

# Flags
window = meta.window
horizon = meta.horizon
dimensionality = meta.dimensionality
n_layers = 2
start_size = 256
optimizer = k.optimizers.get(meta.optimizer)
learning_rate = meta.learning_rate
dev = True

if dev:
    epochs = 2
else:
    epochs = meta.epochs

if dimensionality < 125:
    reduce_dimensions = True
else:
    reduce_dimensions = False

_, train_y, _, test_y, train_generator, test_generator = aursad.get_dataset_generator(path=meta.data_path,
                                                                                      window_size=window,
                                                                                      reduce_dimensionality=reduce_dimensions,
                                                                                      n_dimensions=dimensionality,
                                                                                      loosening_samples=0,
                                                                                      move_samples=0,
                                                                                      batch_size=meta.batch_size)

clf = LSTM_Model(window=window,
                 dimensions=dimensionality,
                 n_layers=n_layers,
                 start_size=start_size,
                 classes=len(np.unique(train_y)),
                 optimizer=optimizer)

clf.train(train_generator=train_generator,
          epochs=epochs)

train_metrics = clf.score(test_labels=train_y,
                          test_generator=train_generator)

test_metrics = clf.score(test_labels=test_y,
                         test_generator=test_generator)

print(f"Train f1_avg: {train_metrics['f1_avg']}")
print(f"Test f1_avg: {test_metrics['f1_avg']}")
