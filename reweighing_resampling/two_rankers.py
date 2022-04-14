import sys
sys.path.append("../")
import numpy as np


if __name__ == '__main__':
    prob_origin = np.load('ranker_result_origin/2dims_result.npy')
    x_origin = np.load('../data/census/data-x.npy')
    y_origin = np.load('../data/census/data-y.npy')
    print('end')
