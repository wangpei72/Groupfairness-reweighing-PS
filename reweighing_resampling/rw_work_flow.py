import sys

sys.path.append("../")
import numpy as np
from aif360.metrics.classification_metric import ClassificationMetric
from aif360.datasets.binary_label_dataset import BinaryLabelDataset



if __name__ == '__main__':
    # 先拿到训练好的模型 和原始的训练集作为输入
    # 1 根据train集的分布计算出四个权重值 in features.npy 2d-labels.npy
    # 2 将train集x 和y进行模型的
