import os

# Tensorflow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import meta as meta
import tensorflow.keras as k

from TABL.tabl_model import TABLModel
import aursad
import json
import numpy as np

# Flags
window = meta.window
horizon = meta.horizon
dimensionality = meta.dimensionality
screwdiver_only = False
binarize = False
optimizer = k.optimizers.get(meta.optimizer)
learning_rate = meta.learning_rate
dropout = 0.1
projection_regularizer = None
projection_constraint = None
attention_regularizer = None
attention_constraint = None
n_bl_layers = 2
bl_dimensions = {0: "[120, 5]", 1: "[60, 2]"}
n_tabl_layers = 1
tabl_dimensions = {0: "[2, 1]"}
dev = False
remote = True

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

if screwdiver_only:
    dimensionality = 7

# Turn dicts of string lists to dict of lists
for k, v in bl_dimensions.items():
    bl_dimensions[k] = json.loads(v)

for k, v in tabl_dimensions.items():
    tabl_dimensions[k] = json.loads(v)

projection_regularizer = None if projection_regularizer == 'None' else projection_regularizer
projection_constraint = None if projection_constraint == 'None' else eval(
    f"tf.keras.constraints.{projection_constraint['name']}({projection_constraint['max_value'], projection_constraint['axis']})")

attention_regularizer = None if attention_regularizer == 'None' else attention_regularizer
attention_constraint = None if attention_constraint == 'None' else eval(
    f"tf.keras.constraints.{attention_constraint['name']}({attention_constraint['max_value'], attention_constraint['axis']})")

_, train_y, _, test_y, train_generator, test_generator = aursad.get_dataset_generator(path=data_path,
                                                                                      window_size=window,
                                                                                      reduce_dimensionality=reduce_dimensions,
                                                                                      binary_labels=binarize,
                                                                                      screwdriver_only=screwdiver_only,
                                                                                      n_dimensions=dimensionality,
                                                                                      normal_samples=normal_samples,
                                                                                      missing_samples=missing_samples,
                                                                                      damaged_samples=damaged_samples,
                                                                                      loosening_samples=0,
                                                                                      move_samples=0,
                                                                                      batch_size=meta.batch_size)

clf = TABLModel(window=window,
                dimensions=dimensionality,
                classes=len(np.unique(train_y)),
                optimizer=optimizer,
                dropout=dropout,
                projection_regularizer=projection_regularizer,
                projection_constraint=projection_constraint,
                attention_regularizer=attention_regularizer,
                attention_constraint=attention_constraint,
                n_bl_layers=n_bl_layers,
                bl_layers=bl_dimensions,
                n_tabl_layers=n_tabl_layers,
                tabl_layers=tabl_dimensions)

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

# 'attention_constraint':
#   description: Attention constraint
#   default: {'name': 'max_norm', 'max_value': 5.0, 'axis' :1}
#
# 'projection_constraint':
# description: Projection
# constraint
# default: {'name': 'max_norm', 'max_value': 3.0, axis: 0}
