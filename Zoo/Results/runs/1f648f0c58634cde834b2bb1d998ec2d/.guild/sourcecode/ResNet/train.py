import os

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import meta as meta
import tensorflow.keras as k

from resnet_model import ResNetModel
import aursad
import numpy as np

# Flags
window = meta.window
horizon = meta.horizon
dimensionality = meta.dimensionality
binarize = False
prediction_mode = True
optimizer = k.optimizers.get(meta.optimizer)
learning_rate = meta.learning_rate
n_feature_maps = 64
dev = False
remote = False

if remote:
    data_path = '~/Data/AURSAD.h5'
else:
    data_path = meta.data_path

# Allow memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, enable=True)

if dev:
    epochs = 2
    normal_samples = 0.1
    missing_samples = 0.1
    damaged_samples = 0.1
else:
    epochs = meta.epochs
    normal_samples = 1
    missing_samples = 1
    damaged_samples = 1

if dimensionality < 125:
    reduce_dimensions = True
else:
    reduce_dimensions = False

_, train_y, _, test_y, train_generator, test_generator = aursad.get_dataset_generator(path=data_path,
                                                                                      window_size=window,
                                                                                      reduce_dimensionality=reduce_dimensions,
                                                                                      n_dimensions=dimensionality,
                                                                                      binary_labels=binarize,
                                                                                      prediction_mode=prediction_mode,
                                                                                      normal_samples=normal_samples,
                                                                                      missing_samples=missing_samples,
                                                                                      damaged_samples=damaged_samples,
                                                                                      loosening_samples=0,
                                                                                      move_samples=0,
                                                                                      batch_size=meta.batch_size)

clf = ResNetModel(window=window,
                  dimensions=dimensionality,
                  classes=len(np.unique(train_y)),
                  optimizer=optimizer,
                  n_feature_maps=n_feature_maps)

clf.train(train_generator=train_generator,
          epochs=epochs)

train_metrics = clf.score(test_labels=train_y,
                          test_generator=train_generator,
                          mode='train')

test_metrics = clf.score(test_labels=test_y,
                         test_generator=test_generator,
                         mode='test')

print(f"Train f1_avg: {train_metrics['f1_avg']}")
print(f"Test f1_avg: {test_metrics['f1_avg']}")
