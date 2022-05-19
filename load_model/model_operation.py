import numpy as np
import sys
sys.path.append("../")

import tensorflow as tf
from tensorflow.python.platform import flags

from utils.utils_tf import *
from tutorial_models import dnn

FLAGS = flags.FLAGS

# data_set_list = ['adult', 'compas', 'german']
# data_shape_list = [(None, 18), (None, 10), (None, 11)]
# data_path_list = ['../data/npy_data_from_aif360/adult-aif360preproc/',
#                   '../data/npy_data_from_aif360/compas-aif360preproc/',
#                   '../data/npy_data_from_aif360/german-aif360preproc/']
# data_set_list = ['bank']
# data_shape_list = [(None, 28)]
# data_path_list = ['../data/npy_data_from_aif360/bank-aif360preproc/']
data_set_list = ['default']
data_shape_list = [(None, 26)]
data_path_list = ['../data/npy_data_from_aif360/default-aif360preproc/']
def training(dataset, model_path, nb_epochs, batch_size,learning_rate,
             dataset_path='../data/npy_data_from_aif360/adult-aif360preproc/',
             input_shape=(None, 18),
             nb_classes=2
             ):
    # prepare the data and model
    X = np.load(dataset_path + "features-train.npy")
    Y = np.load(dataset_path + "2d-labels-train.npy")
    input_shape = input_shape
    nb_classes = nb_classes

    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)

    # training parameters
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': model_path + dataset + "/",
        'filename': 'test.model'
    }

    # training procedure
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2019, 7, 15])
    model_train(sess, x, y, preds, X, Y, args=train_params,
                rng=rng, save=True)

    # evaluate the accuracy of trained model
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    sess.close()


def main(argv=None):

    for i in range(len(data_set_list)):
        # if 'session' in locals() and sess is not None:
        #     sess.close()

        training(dataset = data_set_list[i],
                model_path = FLAGS.model_path,
                nb_epochs=FLAGS.nb_epochs,
                batch_size=FLAGS.batch_size,
                learning_rate=FLAGS.learning_rate,
                dataset_path= data_path_list[i],
                input_shape= data_shape_list[i]
                 )
    print('bank\'s original models trained with dataset bank marketing from aif360 done.')

if __name__ == '__main__':
    # flags.DEFINE_string("dataset", "adult", "the name of dataset")
    flags.DEFINE_string("model_path", "../model_from_aif360data/", "the name of path for saving model")
    flags.DEFINE_integer('nb_epochs', 1000, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')

    tf.app.run()