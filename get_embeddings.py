import os
import os.path
import json
import numpy as np
import sys
import importlib
import tensorflow as tf
import string
import re
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))

MODEL = importlib.import_module('model')

import part_dataset
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model', help='Model name [default: model]')
FLAGS = parser.parse_args()
MODEL = FLAGS.model

NUM_POINT = 2048

category = 'Chair'

DATA_PATH = os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0')
TRAIN_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=category, split='trainval')

length = len(TRAIN_DATASET)

GPU_INDEX = 0

def get_model(MODEL_PATH, batch_size=1, num_point=2048):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            embedding = tf.placeholder(tf.float32, shape=(1, 1024))
            net = tf_util.fully_connected(embedding, 1024, bn=True, is_training=False, scope='fc1', bn_decay=None)
            net = tf_util.fully_connected(net, 1024, bn=True, is_training=False, scope='fc2', bn_decay=None)
            net = tf_util.fully_connected(net, num_point * 3, activation_fn=None, scope='fc3')
            net = tf.reshape(net, (batch_size, num_point, 3))
            saver = tf.train.Saver()
        # Create a session
        print(net)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'embedding': embedding,
               'pred': pred,
               'loss': loss}
        return sess, ops, net


point_clouds = np.array([TRAIN_DATASET[i][0] for i in range(length)])

sess, ops, decoder = get_model(MODEL)

for i in range(0, length, 100):
    curr_100 = []
    for j in range(i, i + 100):
        embedding = sess.run(ops['loss'][1]['embedding'], feed_dict={ops['pointclouds_pl']: point_clouds[i: i + 1], ops['is_training_pl']: False})
        curr_100.append(embedding)
    embeddings = np.array(curr_100)
    np.save('embeddings//embedings_{}_{}.npy'.format(MODEL, i), embeddings)


