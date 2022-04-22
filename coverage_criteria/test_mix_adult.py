#Ma, Lei, et al. "DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems." (2018):120-131.#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import math

sys.path.append("../")


from coverage_criteria.utils import neuron_boundary, calculate_layers, update_multi_coverage_neuron, calculate_coverage_layer, init_coverage_metric, get_single_sample_from_instances_set

from load_model.network import *
from load_model.layer import *

FLAGS = flags.FLAGS

def dnn5(input_shape=(None, 13), nb_classes=2):
    """
    The implementation of a DNN model
    :param input_shape: the shape of dataset
    :param nb_classes: the number of classes
    :return: a DNN model
    """
    activation = ReLU
    layers = [Linear(64),
              activation(),
              Linear(32),
              activation(),
              Linear(16),
              activation(),
              Linear(8),
              activation(),
              Linear(4),
              activation(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model


def model_load(datasets):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.allow_growth = True
    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    input_shape = (None, 13)
    nb_classes = 2
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    feed_dict = None

    model = dnn5(input_shape, nb_classes)

    preds = model(x)
    print("Defined TensorFlow model graph.")

    saver = tf.train.Saver()

    model_path = '../mod/' +datasets+ '/test.model'

    saver.restore(sess, model_path)

    return sess, preds, x, y, model, feed_dict

def multi_testing_criteria(idx_in_range20, id_list_cnt, datasets, model_name, samples_path, std_range = 0.0, k_n = 1000, k_l = 2):
    """
    :param datasets
    :param model
    :param samples_path
    :param std_range
    :param k_n
    :param k_l
    :return:
    """
    # m = np.load('../data/adult/data-x.npy')
    # n = np.load('../data/adult/data-y.npy')

    tuple_res = get_single_sample_from_instances_set(idx_in_range_20=idx_in_range20, id_list_cnt=id_list_cnt)
    samples = tuple_res[2]
    X_train_boundary = tuple_res[0]
    store_path = "../multi_testing_criteria/dnn5/adult/"


    if not os.path.exists(store_path):
        os.makedirs(store_path)
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets)
        boundary = neuron_boundary(sess, x, X_train_boundary, model, feed_dict)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()
        np.save(store_path + "boundary.npy", np.asarray(boundary))
    else:
        boundary = np.load(store_path + "boundary.npy", allow_pickle=True).tolist()

    k_coverage, boundary_coverage, neuron_number = init_coverage_metric(boundary, k_n)

    if samples_path == 'test':
        store_path = store_path + 'test/'
    else:
        store_path = store_path + samples_path.split('/')[-3] + '/'

    if not os.path.exists(store_path):
        cal = True
        os.makedirs(store_path)
    else:
        cal = False

    NP = []
    n_batches = 1

    for num in range(n_batches):
        print(num)
        start = 0
        end = len(samples)
        if cal:
            input_data = samples[start:end]
            tf.reset_default_graph()
            sess, preds, x, y, model, feed_dict = model_load(datasets=datasets)
            layers_output = calculate_layers(sess, x, model, feed_dict, input_data, store_path, num)

            sess.close()
            del sess, preds, x, y, model, feed_dict, input_data
            gc.collect()
        else:
            layers_output = np.load(store_path + 'layers_output_' + str(num) + '.npy', allow_pickle=True)

        k_coverage, boundary_coverage = update_multi_coverage_neuron(layers_output, k_n, boundary, k_coverage, boundary_coverage, std_range)

        layer_coverage = calculate_coverage_layer(layers_output, k_l, end - start)

        if num == 0:
            layer = [set([])] * layer_coverage.shape[0]
        for i in range(len(layer_coverage)):
            for j in range(len(layer_coverage[i])):
                layer[i] = layer[i] | layer_coverage[i][j]

        sample_coverage = np.transpose(layer_coverage, (1, 0))
        for i in range(len(sample_coverage)):
            sc = sample_coverage[i].tolist()
            if sc not in NP:
                NP.append(sc)

        del layers_output
        gc.collect()

    KMN = 0
    NB = 0
    SNA = 0
    for i in range(len(k_coverage)):
        for j in range(len(k_coverage[i])):
            for t in range(len(k_coverage[i][j])):
                if k_coverage[i][j][t] > 0:
                    KMN += 1
            if boundary_coverage[i][j][1] > 0:
                NB += 1
                SNA += 1
            if boundary_coverage[i][j][0] > 0:
                NB += 1
    KMN = 1.0 * KMN / (k_n * neuron_number)
    NB = 1.0 * NB / (2 * neuron_number)
    SNA = 1.0 * SNA / neuron_number

    TKNC = sum(len(neurons) for neurons in layer)
    TKNC = 1.0 * TKNC / neuron_number

    TKNP = len(NP)

    print([KMN, NB, SNA, TKNC, TKNP])
    return [KMN, NB, SNA, TKNC, TKNP]

def main(argv=None):
    idx_in_range_20 = 0
    id_list_cnt = 0
    while id_list_cnt < 1:
        multi_nc_to_save = []
        while idx_in_range_20 < 20:
            multi_nc_to_save.append(multi_testing_criteria(idx_in_range_20, id_list_cnt,
                           datasets = FLAGS.datasets,
                           model_name=FLAGS.model,
                           samples_path=FLAGS.samples,
                           std_range = FLAGS.std_range,
                           k_n = FLAGS.k_n,
                           k_l = FLAGS.k_l))
            idx_in_range_20 += 1
        multi_nc_to_save = np.array(multi_nc_to_save, dtype=np.float64)
        np.save('/multi_testing_criteria/dnn5/adult/' + '20-tests-0' + str(id_list_cnt + 1) + '.npy', multi_nc_to_save)
        id_list_cnt += 1


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'adult', 'The target datasets.')
    flags.DEFINE_string('model', 'dnn5', 'The name of model')
    flags.DEFINE_string('samples', 'test', 'The path to load samples.')  # '../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_float('std_range', 0.0, 'The parameter to difine boundary with std')
    flags.DEFINE_integer('k_n', 1000, 'The number of sections for neuron output')
    flags.DEFINE_integer('k_l', 2, 'The number of top-k neurons in one layer')

    tf.app.run()
