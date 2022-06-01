import os
import sys
sys.path.append("../")

import common_utils as cu
from load_model.util_functions import *
from load_model.model_operation import dataset_list, get_data_path_list, get_data_shape_list


# predict
dataset_name_list = dataset_list()
dataset_path_list = get_data_path_list()
dataset_shape_list = get_data_shape_list()



def learn_all_rankers():
    for i in range(len(dataset_name_list)):
        if i == 0:
            continue
        print('current dataset is %s' % dataset_name_list[i])
        input_shape = dataset_shape_list[i]
        nb_classes = 2
        tf.set_random_seed(1234)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))
        model = dnn(input_shape, nb_classes)
        preds = model(x)
        # construct the gradient graph
        grad_0 = gradient_graph(x, preds)
        model_path = '../model_from_aif360data/' + dataset_name_list[i] + '/dnn5/999/test.model'
        saver = tf.train.import_meta_graph(model_path + '.meta')

        saver.restore(sess, model_path)
        sess.run(
            [tf.global_variables_initializer(),
             tf.local_variables_initializer()]
        )

        # learn ranker
        x_path = dataset_path_list[i] + 'features-train.npy'
        y_path = dataset_path_list[i] + '2d-labels-train.npy'
        x_origin = np.load(x_path)
        y_origin = np.load(y_path)
        ranker_array = np.ndarray((x_origin.shape[0], 2), dtype=np.float32)
        for j in range(x_origin.shape[0]):
            # label_tmp = model_argmax(sess, x, preds, np.array([x_origin[i]]))
            ranker_score = model_probab(sess, x, preds, np.array([x_origin[j]]))
            ranker_array[j] = ranker_score

        #    save result
        save_path = 'ranker_result_origin/' + dataset_name_list[i] + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save('ranker_result_origin/' + dataset_name_list[i] + '/'+ '2dims_result.npy', ranker_array)
        sess.close()
        tf.reset_default_graph()
    print('done with learn all rankers')


if __name__ == '__main__':
    learn_all_rankers()
    print('end')

