import os

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# Use mixed precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

# Allow memory growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import meta as meta
import tensorflow.keras as k

from resnet_model import ResNet_Model
import aursad
import numpy as np

# Flags
window = meta.window
horizon = meta.horizon
dimensionality = meta.dimensionality
binarize = False
optimizer = k.optimizers.get(meta.optimizer)
learning_rate = meta.learning_rate
n_feature_maps = 64
dev = False
remote = False

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

if not remote:
    data_path = meta.data_path
else:
    data_path = '~/Data/AURSAD.h5'


_, train_y, _, test_y, train_generator, test_generator = aursad.get_dataset_generator(path=data_path,
                                                                                      window_size=window,
                                                                                      reduce_dimensionality=reduce_dimensions,
                                                                                      n_dimensions=dimensionality,
                                                                                      binary_labels=binarize,
                                                                                      normal_samples=normal_samples,
                                                                                      missing_samples=missing_samples,
                                                                                      damaged_samples=damaged_samples,
                                                                                      loosening_samples=0,
                                                                                      move_samples=0,
                                                                                      batch_size=meta.batch_size)

clf = ResNet_Model(window=window,
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
