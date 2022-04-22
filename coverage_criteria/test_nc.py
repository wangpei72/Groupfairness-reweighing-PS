# Pei, Kexin, et al. "DeepXplore: Automated Whitebox Testing of Deep Learning Systems." (2017).#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from coverage_criteria import wrt_xls



from load_model.network import *
from load_model.layer import *

sys.path.append("../")


from coverage_criteria.utils import init_coverage_tables, neuron_covered, update_coverage, bank_get_single_sample_from_instances_set



FLAGS = flags.FLAGS


def dnn5(input_shape=(None, 20), nb_classes=2):
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

    input_shape = (None, 20)
    nb_classes = 2
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    feed_dict = None

    model = dnn5(input_shape, nb_classes)

    preds = model(x)
    print("Defined TensorFlow model graph.")

    saver = tf.train.Saver()

    model_path = '../mod/' + datasets + '/test.model'

    saver.restore(sess, model_path)

    return sess, preds, x, y, model, feed_dict


def neuron_coverage(idx_in_range_20, id_list_cnt,datasets, model_name, de=False, attack='fgsm'):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    tuple_res = bank_get_single_sample_from_instances_set(idx_in_range_20, id_list_cnt)
    samples = tuple_res[2]

    n_batches = 10
    X_train_boundary = tuple_res[0]
    store_path = "../multi_testing_criteria/dnn5/bank-additional/"

    for i in range(n_batches):
        print(i)

        tf.reset_default_graph()

        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets)
        model_layer_dict = init_coverage_tables(model)
        model_layer_dict = update_coverage(sess, x, samples, model, model_layer_dict, feed_dict, threshold=0)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()

        result = neuron_covered(model_layer_dict)[2]
        print('covered neurons percentage %d neurons %f'
              % (len(model_layer_dict), result))
        return result


def main(argv=None):
    idx_in_range_20 = 0
    id_list_cnt = 0
    while id_list_cnt < 5:
        nc_to_save = []
        idx_in_range_20 = 0
        while idx_in_range_20 < 20:
            nc_to_save.append(neuron_coverage(idx_in_range_20, id_list_cnt, datasets=FLAGS.datasets,
                                              model_name=FLAGS.model,
                                              ))
            idx_in_range_20 += 1
        nc_to_save = np.array(nc_to_save, dtype=np.float64)
        np.save('../bank-res-nc/20_tests_0' + str(id_list_cnt + 1) + '.npy', nc_to_save)
        id_list_cnt += 1
    id_list_cnt = 0
    while id_list_cnt < 5:
        wrt_xls.wrt_xls_file('bank_neuron_coverage.xls', 'neuron_coverage', 'neuron coverage',
                             path_prefix='../bank-res-nc', id_list_cnt=id_list_cnt)
        id_list_cnt += 1


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'bank-additional', 'The target datasets.')
    flags.DEFINE_string('model', 'dnn5', 'The name of model')


    tf.app.run()
