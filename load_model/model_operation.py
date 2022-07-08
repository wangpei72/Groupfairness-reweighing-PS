import numpy as np
import sys
sys.path.append("../")

import tensorflow as tf
from tensorflow.python.platform import flags

from utils.utils_tf import *
from load_model.tutorial_models import *

FLAGS = flags.FLAGS

data_set_list = ['adult','bank', 'compas', 'default',
                 'german', 'heart', 'meps15', 'meps16',
                 'student']
data_shape_list = [(None, 14), (None, 20), (None, 21), (None, 23),
                   (None, 20), (None, 13), (None, 42), (None, 42),
                   (None, 32)]
data_path_list = ['../data/npy_data_from_aif360/adult-aif360preproc/',
                  '../data/npy_data_from_aif360/bank-aif360preproc/',
                  '../data/npy_data_from_aif360/compas-aif360preproc/',
                  '../data/npy_data_from_aif360/default-aif360preproc/',
                  '../data/npy_data_from_aif360/german-aif360preproc/',
                  '../data/npy_data_from_aif360/heart-aif360preproc/',
                  '../data/npy_data_from_aif360/meps15-aif360preproc/',
                  '../data/npy_data_from_aif360/meps16-aif360preproc/',
                  '../data/npy_data_from_aif360/student-aif360preproc/'
                  ]
data_path_with_d_list = ['../data/result_dataset_rename/adult_race/',
                  '../data/result_dataset_rename/adult_sex/',
                  '../data/result_dataset_rename/compas_race/',
                  '../data/result_dataset_rename/compas_sex/',
                  '../data/result_dataset_rename/german_sex/',
                  '../data/result_dataset_rename/bank_age/',
                  '../data/result_dataset_rename/default_sex/',
                  '../data/result_dataset_rename/heart_age/',
                  '../data/result_dataset_rename/student_sex/',
                  '../data/result_dataset_rename/meps15_race/',
                  '../data/result_dataset_rename/meps16_race/'
                  ]
dataset_with_d_attr_list = ['adult_race', 'adult_sex',
                            'compas_race', 'compas_sex',
                            'german_sex',
                            'bank_age',
                            'default_sex',
                            'heart_age',
                            'student_sex',
                            'meps15_race',
                            'meps16_race']
data_with_d_shape_list = [(None, 14),(None, 14),
                          (None, 21), (None, 21),
                          (None, 20),
                          (None, 20),
                          (None, 23),
                          (None, 13),
                          (None, 32),
                          (None, 42),
                          (None, 42)
                   ]
def dataset_list():
    return data_set_list

def get_data_shape_list():
    return data_shape_list

def get_data_path_list():
    return data_path_list

def training(dataset, model_path, nb_epochs, batch_size,learning_rate,
             dataset_path='../data/npy_data_from_aif360/adult-aif360preproc/',
             input_shape=(None, 14),
             nb_classes=2,
             model_type='dnn5'
             ):
    X = np.load(dataset_path + model_type + "/features-train.npy")
    Y = np.load(dataset_path + model_type + "/2d-labels-train.npy")
    input_shape = input_shape
    nb_classes = nb_classes
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    if model_type == 'dnn1':
        model = dnn1(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn3':
        model = dnn3(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn7':
        model = dnn7(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn9':
        model = dnn9(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn2':
        model = dnn2(input_shape, nb_classes)
        preds = model(x)
    elif model_type == 'dnn4':
        model = dnn4(input_shape, nb_classes)
        preds = model(x)
    else:
        assert model_type == 'dnn5'
        model = dnn5(input_shape, nb_classes)
        preds = model(x)

    # training parameters
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': model_path + dataset + "/" + model_type + '/',
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

    # for i in range(len(data_set_list)):
    #     # if 'session' in locals() and sess is not None:
    #     #     sess.close()
    #     if i == 0:
    #         continue
    #     training(dataset = data_set_list[i],
    #             model_path = FLAGS.model_path,
    #             nb_epochs=FLAGS.nb_epochs,
    #             batch_size=FLAGS.batch_size,
    #             learning_rate=FLAGS.learning_rate,
    #             dataset_path= data_path_list[i],
    #             input_shape= data_shape_list[i]
    #              )
    # print('%s\'s original models trained with dataset home from aif360 done.' % data_set_list[i])
    model_type_list = ['dnn1', 'dnn2', 'dnn3', 'dnn4', 'dnn5']
    for j in range(len(model_type_list)):
        model_type = model_type_list[j]
        print('=======current model type is %s=======' % model_type)
        for i in range(len(dataset_with_d_attr_list)):
            print('>>>>>current dataset is %s id %d' % (dataset_with_d_attr_list[i], i+1))
            training(dataset=dataset_with_d_attr_list[i],
                     model_type=model_type,
                     model_path=FLAGS.model_path,
                     nb_epochs=FLAGS.nb_epochs,
                     batch_size=FLAGS.batch_size,
                     learning_rate=FLAGS.learning_rate,
                     dataset_path=data_path_with_d_list[i],
                     input_shape=data_with_d_shape_list[i]
                     )
            print('retrain model based on gen dataset %s done. ' % dataset_with_d_attr_list[i])
    print('reweighing 11 models trained with gen datasets done, ready to run nc and sa.')


if __name__ == '__main__':
    # flags.DEFINE_string("dataset", "adult", "the name of dataset")
    flags.DEFINE_string("model_path", "../model_reweighing/", "the name of path for saving model")
    flags.DEFINE_integer('nb_epochs', 1000, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')

    tf.app.run()