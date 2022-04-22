import sys

sys.path.append("../")
import numpy as np
import time
from load_model.util_functions import *
from group_fairness_metric import calcu_all as ca

input_shape = (None, 13)
nb_classes = 2
tf.set_random_seed(1234)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
x = tf.placeholder(tf.float32, shape=input_shape)
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
model = dnn(input_shape, nb_classes)
preds = model(x)
saver = tf.train.Saver()
# model_path = '../mod/adult/my-model/census/999/' + 'test.model'
model_path = '../mod/adult/start-all-over-model/adult/999/' + 'test.model'
saver.restore(sess, model_path)

# construct the gradient graph
grad_0 = gradient_graph(x, preds)
# predict

if __name__ == '__main__':
    x_origin = np.load('../data/adult/data-x.npy')
    y_origin = np.load('../data/adult/data-y.npy')
    id_list =['05']
    for i in range(1):
        test_instance_array = np.load('../data/test-set/test_instances_set' +id_list[i] + '.npy', allow_pickle=True)
        test_res_accurate = []
        test_accuracy = []
        accuracy = 0.
        cor_cnt = 0
        wro_cnt = 0
        ground_truth_tmp = 0
        eood = 0.
        eoop = 0.
        sample_id = 1
        y_pre = []
        X_sample = []
        y_true = []
        for idx in test_instance_array[19]:
            test_res_tmp = []
            sample_tmp = x_origin[idx]
            label_tmp = model_argmax(sess, x, preds, np.array([sample_tmp]))
            X_sample.append(sample_tmp)  # 保存当前的instance
            y_pre.append(label_tmp)  # 保存当前推理获得的y值 0 :<50k 1: >50k
            ground_truth_tmp_array = y_origin[idx]
            if ground_truth_tmp_array[0] > 0:
                ground_truth_tmp = 0
            else:
                ground_truth_tmp = 1
            y_true.append(ground_truth_tmp)
            if label_tmp == ground_truth_tmp:
                test_res_tmp.append(1)
                cor_cnt += 1
                # print("sample id: %d correct prediction, record 1 for this sample" % sample_id)
            else:
                test_res_tmp.append(0)
                wro_cnt += 1
                # print("sample id: %d wrong prediction, record 0 for this sample" % sample_id)
            sample_id += 1
        X_arr = np.array(X_sample, dtype=np.float32)
        y_arr = np.array(y_pre, dtype=np.float32)
        y_true_arr = np.array(y_true, dtype=np.float32)

        spd, di, eoop, eood = ca.calcu_all_metrics(X_arr, y_arr, y_true_arr)

        test_res_accurate.append(test_res_tmp)
        accuracy = cor_cnt / (cor_cnt + wro_cnt)
        test_accuracy.append(accuracy)
        print('test accuracy i %f' % accuracy)
        print('sample nums is %d' % sample_id)
        print('group fairness of predicted test set by new model is:')
        print('SPD : %f' % spd)
        print('DI : %f' % di)
        print('EOOP : %f' % eoop)
        print('EOOD : %f' % eood)
        array_to_save = np.array([sample_id, accuracy, spd, di, eoop, eood])
        np.save('./accu_gf_metrics/new_model.npy', array_to_save)

