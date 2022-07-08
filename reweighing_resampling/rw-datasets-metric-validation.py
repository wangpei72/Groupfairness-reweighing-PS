import os
import sys
sys.path.append("../")
import numpy as np
from aif360.algorithms.preprocessing.optim_preproc_helpers.structure_dataset_helper.metric_work_flow import *


model_type_list = ['dnn1', 'dnn2', 'dnn3', 'dnn4', 'dnn5']


if __name__ == '__main__':
    for j in range(len(model_type_list)):
        model_type = model_type_list[j]
        for i in range(len(data_set_list_compat)):
            dataset_name = data_set_list_compat[i]
            dataset_name_d = dataset_with_d_attr_list[i]
            print('==============%s================' % dataset_name_d)
            print('>>>>>>>>>>DNN%d<<<<<<<<<<<' % (j + 1))
            x_path = 'result_dataset_rename/' + dataset_name_d + '/' + model_type + '/features-train.npy'
            y_path = 'result_dataset_rename/' + dataset_name_d + '/' + model_type + '/2d-labels-train.npy'
            x = np.load(x_path)
            y = np.load(y_path)
            y_convert_1d = np.argmax(y, axis=1)
            dm = structring_binary_datasetmetric_from_npy_array(x, y_convert_1d,
                                                                dataset_name=dataset_name,
                                                                dataset_with_d_name=dataset_name_d,
                                                                print_bool=True)

    print('done')
