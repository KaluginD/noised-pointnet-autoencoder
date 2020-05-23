import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))

import part_dataset

import tf_util

import tf_nndistance


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)

NUM_POINT = 2048

category = 'Chair'

sess = tf.Session(config=config)

DATA_PATH = os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0')
TRAIN_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=category, split='trainval')


def get_loss(pred, label):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    loss = tf.reduce_mean(dists_forward+dists_backward, axis=1)
    return loss*100


length = len(TRAIN_DATASET)

for i in range(length):
    first = np.array([TRAIN_DATASET[i][0]] * length)
    second = np.array([TRAIN_DATASET[j][0] for j in range(length)])
    distances = get_loss(first, second)
    result = sess.run(distances)
    np.save('distances/dists_{}.npy'.format(i), result)

pairs = [(i, j) for i in range(100) for j in range(100)]

first = np.array([TRAIN_DATASET[pair[0]] for pair in pairs])
second = np.array([TRAIN_DATASET[pair[1]] for pair in pairs])

distances = get_loss(first, second)
np.save('distances.npy', distances)
