import sys

sys.path.append("../")
import numpy as np
from aif360.metrics.classification_metric import ClassificationMetric
from aif360.datasets.binary_label_dataset import BinaryLabelDataset

if __name__ == '__main__':
    # 训练好的模型
    # 1 根据train集的分布计算出四个权重值
    # 2 将train集x 和y进行模型的