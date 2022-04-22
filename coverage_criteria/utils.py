from collections import defaultdict
import numpy as np
from six.moves import xrange
from scipy.special import comb
import tensorflow as tf
from itertools import combinations
import sys
sys.path.append("../")


def init_coverage_tables(model):
    model_layer_dict = defaultdict(bool)
    for layer in model.layers:
        if 'Flatten' in layer.name or 'Input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False
    # init_dict(model, model_layer_dict)
    return model_layer_dict

# def init_dict(model, model_layer_dict):
#     for layer in model.layers:
#         if 'Flatten' in layer.name or 'Input' in layer.name:
#             continue
#         for index in range(layer.output_shape[-1]):
#             model_layer_dict[(layer.name, index)] = False

def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

# def update_coverage(sess, x, input_data, model, model_layer_dict, feed_dict, threshold=0):
#     layer_names = [layer.name for layer in model.layers if
#                    'Flatten' not in layer.name and 'Input' not in layer.name]
#     intermediate_layer_outputs = []
#     dict = model.fprop(x)
#     for key in model.layer_names:
#         if 'Flatten' not in key and 'Input' not in key:
#             tensor = dict[key]
#             feed = {x: input_data}
#             if feed_dict is not None:
#                 feed.update(feed_dict)
#             v = sess.run(tensor, feed_dict=feed)
#             intermediate_layer_outputs.append(v)
#     del v
#
#     for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
#         for j in range(len(intermediate_layer_output)):
#             scaled = scale(intermediate_layer_output[j])
#             for num_neuron in xrange(scaled.shape[-1]):
#                 if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
#                     model_layer_dict[(layer_names[i], num_neuron)] = True
#     del intermediate_layer_outputs, intermediate_layer_output
#     return model_layer_dict

def update_coverage(sess, x, input_data, model, model_layer_dict, feed_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'Flatten' not in layer.name and 'Input' not in layer.name]
    intermediate_layer_outputs = []
    dict = model.fprop(x)
    for key in model.layer_names:
        if 'Flatten' not in key and 'Input' not in key:
            tensor = dict[key]
            feed = {x: input_data}
            if feed_dict is not None:
                feed.update(feed_dict)
            layer_output = sess.run(tensor, feed_dict=feed)

            layer_op = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
            for j in range(len(layer_output)):
                scaled = scale(layer_output[j])
                for num_neuron in xrange(scaled.shape[-1]):
                    layer_op[j][num_neuron] = np.mean(scaled[..., num_neuron])
            intermediate_layer_outputs.append(layer_op)
    del layer_output, layer_op, scaled

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        for j in range(len(intermediate_layer_output)):
            intermediate_neuron_output = intermediate_layer_output[j]
            for num_neuron in xrange(len(intermediate_neuron_output)):
                if intermediate_neuron_output[num_neuron] > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                    model_layer_dict[(layer_names[i], num_neuron)] = True
    del intermediate_layer_outputs, intermediate_layer_output, intermediate_neuron_output
    return model_layer_dict

def neuron_boundary(sess, x, input_data, model, feed_dict):
    boundary = []
    dict = model.fprop(x)
    for key in model.layer_names:
        if 'Flatten' not in key and 'Input' not in key:
            tensor = dict[key]
            n_batches = int(np.ceil(1.0 * input_data.shape[0] / 256))
            for i in range(n_batches):
                start = i * 256
                end = np.minimum(len(input_data), (i + 1) * 256)
                feed= {x: input_data[start:end]}
                if feed_dict is not None:
                    feed.update(feed_dict)

                layer_output = sess.run(tensor, feed_dict=feed)
                layer_op = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
                for j in range(len(layer_output)):
                    for num_neuron in xrange(layer_output[j].shape[-1]):
                            layer_op[j][num_neuron] = np.mean(layer_output[j][..., num_neuron])
                layer_op = np.transpose(layer_op, (1, 0))
                low = np.min(layer_op, axis=1)
                high = np.max(layer_op, axis=1)
                mean = np.mean(layer_op, axis=1)
                std = np.std(layer_op, axis=1, ddof=1)
                if i == 0:
                    layer = np.transpose(np.asarray([low, high, std, mean]), (1, 0))
                else:
                    l = np.transpose(np.asarray([low, high, std, mean]), (1, 0))
                    n1 = start
                    n2 = end - start
                    for j in range(len(l)):
                        if l[j][0] < layer[j][0]:
                            layer[j][0] = l[j][0]
                        if l[j][1] > layer[j][1]:
                            layer[j][1] = l[j][1]
                        layer[j][2] = pow(((n1-1)*pow(layer[j][2],2)+(n2-1)*pow(l[j][2],2) + n1*n2*(pow(layer[j][3],2)+pow(l[j][3],2)-2*layer[j][3]*l[j][3])/(1.0*n1+n2)) / (1.0*n1+n2-1), 0.5)
            boundary.append(layer[:,:3])
    return boundary

# def calculate_layers(sess, x, input_data, model, feed_dict):
#     layers_output = []
#     dict = model.fprop(x)
#     for key in model.layer_names:
#         if 'Flatten' not in key and 'Input' not in key:
#             tensor = dict[key]
#             layer_output = []
#             n_batches = int(np.ceil(1.0 * input_data.shape[0] / 32))
#             for i in range(n_batches):
#                 start = i * 32
#                 end = np.minimum(len(input_data), (i + 1) * 32)
#                 feed = {x: input_data[start:end]}
#                 if feed_dict is not None:
#                     feed.update(feed_dict)
#                 v = sess.run(tensor, feed_dict=feed)
#                 layer_output = layer_output + v.tolist()
#
#             layer_output = np.asarray(layer_output)
#             layer_op = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
#             for j in range(len(layer_output)):
#                 for num_neuron in xrange(layer_output[j].shape[-1]):
#                     layer_op[j][num_neuron] = np.mean(layer_output[j][..., num_neuron])
#
#             layer_op = np.transpose(layer_op, (1, 0))  # num_neurons * num_samples
#             layers_output.append(layer_op)
#
#     return np.asarray(layers_output)

# def update_multi_coverage_neuron(layers_output, k, boundary, k_coverage, boundary_coverage, std_range):
#     for i in range(len(layers_output)):
#         for j in range(len(layers_output[i])):
#             lower_bound = boundary[i][j][0] - std_range * boundary[i][j][2]
#             upper_bound = boundary[i][j][1] + std_range * boundary[i][j][2]
#             for t in range(len(layers_output[i][j])):
#                 output = layers_output[i][j][t]
#                 lower = boundary[i][j][0]
#                 upper = boundary[i][j][1]
#                 if output < lower_bound:
#                     boundary_coverage[i][j][0] += 1
#                 elif output > upper_bound:
#                     boundary_coverage[i][j][1] += 1
#                 elif output >= lower and output <= upper:
#                     if output == lower:
#                         k_coverage[i][j][0] += 1
#                     else:
#                         addition = 1.0 * (upper - lower) / k
#                         if addition == 0.0:
#                             k_coverage[i][j] = np.add(np.asarray(k_coverage[i][j]), 1).tolist()
#                         else:
#                             section = int(np.ceil(1.0 * (output - lower) / addition)) - 1
#                             if section >= k:
#                                 section = k - 1
#                             k_coverage[i][j][section] += 1
#     return k_coverage, boundary_coverage

def init_coverage_metric(boundary, k_n):
    size = 0
    k_coverage = []
    boundary_coverage = []
    for i in range(len(boundary)):
        k_coverage.append(np.zeros((len(boundary[i]), k_n)).astype('int').tolist())
        boundary_coverage.append(np.zeros((len(boundary[i]),2)).astype('int').tolist())
        size += len(boundary[i])
    return k_coverage, boundary_coverage, size

def calculate_coverage_layer(layers_output, k_l, samples_num):
    layer_coverage = []
    for i in range(len(layers_output)):
        layer_output = np.transpose(layers_output[i], (1,0)) # num_samples * num_neurons
        for j in range(len(layer_output)):
            layer_coverage.append(topk(layer_output[j], k_l))

    layer_coverage = np.asarray(layer_coverage).reshape((layers_output.shape[0], samples_num))
    del layer_output
    return layer_coverage

def topk(neuron_output, k):
    top = np.argsort(neuron_output)
    return set(top[-k:])

def neuron_combination(t, model, x):
    dict = model.fprop(x)
    l_comb = []
    for key in model.layer_names:
        if 'ReLU' in key:
            tensor = dict[key]
            output_shape = tensor.shape[-1]
            combination = int(comb(int(output_shape), t))
            configuration = pow(2, t)
            l_comb.append(np.zeros((combination, configuration)).astype('int').tolist())
    return l_comb

def cal_activation(sess, x, input_data, model, feed_dict):
    layers_activation = []
    dict = model.fprop(x)
    for key in model.layer_names:
        if 'ReLU' in key:
            tensor = dict[key]
            layer_output = []
            n_batches = int(np.ceil(1.0 * input_data.shape[0] / 512))
            for i in range(n_batches):
                start = i * 512
                end = np.minimum(len(input_data), (i + 1) * 512)
                feed = {x: input_data[start:end]}
                if feed_dict is not None:
                    feed.update(feed_dict)
                v = sess.run(tensor, feed_dict=feed)
                layer_output = layer_output + v.tolist()

            layer_output = np.asarray(layer_output)
            layer_activation = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
            for j in range(len(layer_output)):
                for num_neuron in xrange(layer_output[j].shape[-1]):
                    if np.mean(layer_output[j][..., num_neuron]) > 0.0:
                        layer_activation[j][num_neuron] = 1

            layers_activation.append(layer_activation)
    del layer_output
    return layers_activation

def update_combination(layers_combination, layers_activation, t):
    for i in range(len(layers_activation)): #every layer's samples
        for neurons_activation in layers_activation[i]: #every sample's neurons
            c = list(combinations(range(len(neurons_activation)), t))
            for j in range(len(c)):
                conf = int(sum([neurons_activation[c[j][k]] * pow(2, t-1-k) for k in range(t)]))
                layers_combination[i][j][conf] = 1
    return layers_combination

def at_training(sess, x, input_data, model, feed_dict, lsa_layer):
    if lsa_layer >= 0 and lsa_layer < len(model.layer_names):
        layer_name = model.layer_names[lsa_layer]
    else:
        layer_name = model.layer_names[-3]
    dict = model.fprop(x)
    tensor = dict[layer_name]
    # layer_output = []
    # n_batches = int(np.ceil(1.0 * input_data.shape[0] / 256))
    # for i in range(n_batches):
    #     start = i * 256
    #     end = np.minimum(len(input_data), (i + 1) * 256)
    #     feed = {x: input_data[start:end]}
    #     if feed_dict is not None:
    #         feed.update(feed_dict)
    #     v = sess.run(tensor, feed_dict=feed)
    #     layer_output = layer_output + v.tolist()
    # layer_output = np.asarray(layer_output)
    feed = {x: input_data}
    if feed_dict is not None:
        feed.update(feed_dict)
    layer_output = sess.run(tensor, feed_dict=feed)

    layer_op = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
    for j in range(len(layer_output)):
        for num_neuron in xrange(layer_output[j].shape[-1]):
            layer_op[j][num_neuron] = np.mean(layer_output[j][..., num_neuron])

    del layer_output
    return np.asarray(layer_op).reshape(len(input_data), -1)

def cal_sign(sess, x, input_data, model, feed_dict):
    layers_sign = []
    dict = model.fprop(x)
    for key in model.layer_names:
        if 'ReLU' in key or 'probs' in key:
            tensor = dict[key]
            layer_output = []
            n_batches = int(np.ceil(1.0 * input_data.shape[0] / 32))
            for i in range(n_batches):
                start = i * 32
                end = np.minimum(len(input_data), (i + 1) * 32)
                feed = {x: input_data[start:end]}
                if feed_dict is not None:
                    feed.update(feed_dict)
                v = sess.run(tensor, feed_dict=feed)
                layer_output = layer_output + v.tolist()

            layer_output = np.asarray(layer_output)
            layer_sign = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
            for j in range(len(layer_output)):
                for num_neuron in xrange(layer_output[j].shape[-1]):
                    if np.mean(layer_output[j][..., num_neuron]) > 0.0:
                        layer_sign[j][num_neuron] = 1

            layers_sign.append(layer_sign)

    return layers_sign

def xor(list_a, list_b):
    list = []
    for i in range(len(list_a)):
        if list_a[i] != list_b[i]:
            list.append(i)
    return list

def calculate_layers(sess, x, model, feed_dict, input_data, store_path, num):
    layers_output = []
    dict = model.fprop(x)
    for key in model.layer_names:
        if 'Flatten' not in key and 'Input' not in key:
            tensor = dict[key]
            feed = {x: input_data}
            if feed_dict is not None:
                feed.update(feed_dict)
            layer_output = sess.run(tensor, feed_dict=feed)

            layer_op = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
            for j in range(len(layer_output)):
                for num_neuron in xrange(layer_output[j].shape[-1]):
                    layer_op[j][num_neuron] = np.mean(layer_output[j][..., num_neuron])

            layer_op = np.transpose(layer_op, (1, 0))  # num_neurons * num_samples
            layers_output.append(layer_op)
    layers_output = np.asarray(layers_output)
    np.save(store_path + "layers_output_" + str(num) + ".npy", layers_output)
    del layer_op
    return layers_output

def update_multi_coverage_neuron(layers_output, k_n, boundary, k_coverage, boundary_coverage, std_range):
    for i in range(len(layers_output)):
        for j in range(len(layers_output[i])):
            for t in range(len(layers_output[i][j])):
                if layers_output[i][j][t] < boundary[i][j][0] - std_range * boundary[i][j][2]:
                    boundary_coverage[i][j][0] = boundary_coverage[i][j][0] + 1
                elif layers_output[i][j][t] > boundary[i][j][1] + std_range * boundary[i][j][2]:
                    boundary_coverage[i][j][1] = boundary_coverage[i][j][1] + 1
                elif layers_output[i][j][t] >= boundary[i][j][0] and layers_output[i][j][t] <= boundary[i][j][1]:
                    if layers_output[i][j][t] == boundary[i][j][0]:
                        k_coverage[i][j][0] = k_coverage[i][j][0] + 1
                    else:
                        addition = 1.0 * (boundary[i][j][1] - boundary[i][j][0]) / k_n
                        if addition == 0.0:
                            k_coverage[i][j] = np.add(np.asarray(k_coverage[i][j]), 1).tolist()
                        else:
                            section = int(
                                np.ceil(1.0 * (layers_output[i][j][t] - boundary[i][j][0]) / addition)) - 1
                            if section >= k_n:
                                section = k_n - 1
                            k_coverage[i][j][section] = k_coverage[i][j][section] + 1
    del addition
    return k_coverage, boundary_coverage

def get_single_sample_from_instances_set(idx_in_range_20, id_list_cnt, raw_data_path_prefix='../data/adult/'):
    """
    :param idx_in_range_20:一个instance_set文件读取的array中的列索引，代表20次实验取样的sample序号集合数据，取值范围0-19
    :param id_list_cnt: 5次实验的实验序号，取值范围0-4
    :param raw_data_path_prefix: 原始数据集路径前缀，默认为adult数据集路径
    :return: 训练集和用作测试集的4个numpy ndarray X_train, Y_train, X_test, Y_test
    """
    # Object used to keep track of (and return) key accuracies
    m = np.load(raw_data_path_prefix + 'data-x.npy')
    n = np.load(raw_data_path_prefix + 'data-y.npy')
    p = int(m.shape[0] * 0.8)
    X_train = m[:p]
    Y_train = n[:p]
    X_list = []
    Y_list = []
    # 这个X_test由读取instance-set文件拼凑成 x y，所以func要返回x , y,并且传递id—list-cnt参数
    id_list = ['01', '02', '03', '04', '05']
    test_instances_array = np.load(raw_data_path_prefix + 'test_set/test_instances_set' + id_list[id_list_cnt] + '.npy', allow_pickle=True)
    for idx in test_instances_array[idx_in_range_20]:
        X_list.append(m[idx])
        Y_list.append(n[idx])
    X_test = np.array(X_list, dtype=np.float64)
    Y_test = np.array(Y_list, dtype=np.float64)

    return X_train, Y_train, X_test, Y_test


def bank_get_single_sample_from_instances_set(idx_in_range_20, id_list_cnt, raw_data_path_prefix='../data/bank/'):
    """
    :param idx_in_range_20:一个instance_set文件读取的array中的列索引，代表20次实验取样的sample序号集合数据，取值范围0-19
    :param id_list_cnt: 5次实验的实验序号，取值范围0-4
    :param raw_data_path_prefix: 原始数据集路径前缀，默认为adult数据集路径
    :return: 训练集和用作测试集的4个numpy ndarray X_train, Y_train, X_test, Y_test
    """
    # Object used to keep track of (and return) key accuracies
    m = np.load(raw_data_path_prefix + 'data-bank-additional-X1.npy')
    n = np.load(raw_data_path_prefix + 'data-bank-additional-Y1.npy')
    p = int(m.shape[0] * 0.8)
    X_train = m[:p]
    Y_train = n[:p]
    X_list = []
    Y_list = []
    # 这个X_test由读取instance-set文件拼凑成 x y，所以func要返回x , y,并且传递id—list-cnt参数
    id_list = ['01', '02', '03', '04', '05']
    test_instances_array = np.load(raw_data_path_prefix + 'test_set/bank_test_instances_set' + id_list[id_list_cnt] + '.npy', allow_pickle=True)
    for idx in test_instances_array[idx_in_range_20]:
        X_list.append(m[idx])
        Y_list.append(n[idx])
    X_test = np.array(X_list, dtype=np.float64)
    Y_test = np.array(Y_list, dtype=np.float64)

    return X_train, Y_train, X_test, Y_test

