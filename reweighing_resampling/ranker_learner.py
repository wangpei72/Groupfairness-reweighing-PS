import sys
sys.path.append("../")

import common_utils as cu
from load_model.util_functions import *

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
model_path = '../my-model/adult/999/' + 'test.model'
saver.restore(sess, model_path)

# construct the gradient graph
grad_0 = gradient_graph(x, preds)
# predict
if __name__ == '__main__':
    x_origin = np.load('../data/adult/data-x.npy')
    y_origin = np.load('../data/adult/data-y.npy')
    ranker_array = np.ndarray((x_origin.shape[0], 2), dtype=np.float32)
    print('start predict')
    for i in range(x_origin.shape[0]):
        label_tmp = model_argmax(sess, x, preds, np.array([x_origin[i]]))
        ranker_score = model_probab(sess, x, preds, np.array([x_origin[i]]))
        ranker_array[i] = ranker_score
    np.save('ranker_result_origin/2dims_result.npy', ranker_array)
    print('end')

