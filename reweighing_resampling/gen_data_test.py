import sys

sys.path.append("../")
import numpy as np

if __name__ == '__main__':
    x = np.load('result_dataset/x_generated.npy')
    y = np.load('result_dataset/y_generated.npy')
    test = np.load('../data/adult/train/census-100%-x-train.npy')

    x_train, x_test = np.split(x, indices_or_sections=[26047])
    y_train, y_test = np.split(y, indices_or_sections=[26047])

    np.save('../data/census-gen/train/census-gen-x-train.npy', x_train)
    np.save('../data/census-gen/train/census-gen-y-train.npy', y_train)
    np.save('../data/census-gen/test/census-gen-x-test.npy', x_test)
    np.save('../data/census-gen/test/census-gen-y-test.npy', y_test)
    pass