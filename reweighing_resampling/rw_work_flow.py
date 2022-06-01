import sys
from time import time

sys.path.append("../")
import numpy as np

from two_rankers import gen_all_sets
from ranker_learner import learn_all_rankers
data_set_list = ['adult', 'compas', 'german', 'bank',
                 'default', 'heart', 'student',
                 'meps15', 'meps16']
data_set_list_compat = ['adult', 'adult',
                        'compas', 'compas',
                        'german', 'bank',
                        'default', 'heart', 'student',
                        'meps15', 'meps16']
dataset_with_d_attr_list = ['adult_race', 'adult_sex',
                            'compas_race', 'compas_sex',
                            'german_sex',
                            'bank_age',
                            'default_sex',
                            'heart_age',
                            'student_sex',
                            'meps15_race',
                            'meps16_race']
dataset_d_attr_name_map = {'adult_race': 'race', 'adult_sex': 'sex', 'compas_race': 'race',
                           'compas_sex': 'sex', 'german_sex': 'sex', 'bank_age': 'age',
                           'default_sex': 'x2', 'heart_age': 'age', 'student_sex': 'sex',
                           'meps15_race': 'race', 'meps16_race': 'race'}

d_attr_idx_map = {'adult_race': 8, 'adult_sex': 9, 'compas_race': 2,
                  'compas_sex': 0, 'german_sex': 8, 'bank_age': 0,
                  'default_sex': 1, 'heart_age': 0, 'student_sex': 1,
                  'meps15_race': 3, 'meps16_race': 3}

d_categoric_meta_map = {'adult_race': {1: 'White', 0: 'Non-white'},
                        'adult_sex': {1: 'Male', 0: 'Female'},
                        'compas_race': {1: 'Caucasian', 0: 'Not Caucasian'},
                        'compas_sex': {0: 'Male', 1: 'Female'},
                        'german_sex': {1: 'Male', 0: 'Female'},
                        'bank_age': {1: 'Old', 0: 'Young'},
                        'default_sex': {1: 'Male', 0: 'Female'},
                        'heart_age': {1: 'Young', 0: 'Old'},
                        'student_sex': {1: 'Male', 0: 'Female'},
                        'meps15_race': {1: 'White', 0: 'Non-white'},
                        'meps16_race': {1: 'White', 0: 'Non-white'}}

fav_d_map = {'adult_race': 1,
             'adult_sex': 1,
             'compas_race': 1,
             'compas_sex': 1,
             'german_sex': 1,
             'bank_age': 1,
             'default_sex': 1,
             'heart_age': 1,
             'student_sex': 1,
             'meps15_race': 1,
             'meps16_race': 1}
# TODO 排序后的ranker进行是否有保存的文件供加载的判断
fav_cond_map = {
             'adult_race': [{'race': 1}],
             'adult_sex': [{'sex': 1}],
             'compas_race': [{'race': 1}],
             'compas_sex': [{'sex': 1}],
             'german_sex': [{'sex': 1}],
             'bank_age': [{'age': 3}, {'age': 4},{'age': 5}, {'age': 6},{'age': 7}, {'age': 8},{'age': 9}],
             'default_sex': [{'sex': 1}],
             'heart_age': [ {'age': 2},{'age': 3}],
             'student_sex': [{'sex': 1}],
             'meps15_race': [{'race': 1}],
             'meps16_race': [{'race': 1}]}
unfav_cond_map = {
             'adult_race': [{'race': 0}],
             'adult_sex': [{'sex': 0}],
             'compas_race': [{'race': 0}],
             'compas_sex': [{'sex': 0}],
             'german_sex': [{'sex': 0}],
             'bank_age': [{'age': 0}, {'age': 1}, {'age': 2}],
             'default_sex': [{'sex': 0}],
             'heart_age': [ {'age': 4},{'age': 5}, {'age': 6},{'age': 7}],
             'student_sex': [{'sex': 0}],
             'meps15_race': [{'race': 0}],
             'meps16_race': [{'race': 0}]
}

def para_print(attr, idx, x_path, y_path, condition_p, condition_up):
    print('protected attribute is %s' % attr)
    print('protected attribute idx is %d' % idx)
    print('x path is %s' % x_path)
    print('y path is %s' % y_path)
    print('condition_pri:')
    print(condition_p)
    print('condition_unpri:')
    print(condition_up)


def get_all_reweighing_generated_sets():
    for i in range(len(dataset_with_d_attr_list)):
        if i == 1:
            continue
        dataset_d_name = dataset_with_d_attr_list[i]
        print('=================current dataset and protected_attr is %s==================' % dataset_d_name)
        s_time = time()
        protected_attr = dataset_d_attr_name_map[dataset_d_name]
        protected_attr_idx = d_attr_idx_map[dataset_d_name]
        ranker_file_path = 'ranker_result_origin/' + dataset_d_name.split('_')[0] + '/2dims_result.npy'
        x_path = '../data/npy_data_from_aif360/' + dataset_d_name.split('_')[0] + '-aif360preproc/features-train.npy'
        y_path = '../data/npy_data_from_aif360/' + dataset_d_name.split('_')[0] + '-aif360preproc/2d-labels-train.npy'
        cond_pri = fav_cond_map[dataset_d_name]
        cond_unpri = unfav_cond_map[dataset_d_name]
        para_print(protected_attr, protected_attr_idx, x_path, y_path, cond_pri, cond_unpri)
        gen_all_sets(ranker_filepath=ranker_file_path, x_path=x_path, y_path=y_path,
                     dataset_d_name=dataset_d_name,
                     protected_attribute_names=protected_attr,
                     protected_idx=protected_attr_idx,
                     favorable_label=1,
                     unfavorable_label=0,
                     condition_dict_priv=cond_pri,
                     condition_dict_unpriv=cond_unpri)
        e_time = time()
        dura = e_time - s_time
        print('dataset %s took time : %f s' % (dataset_d_name, dura))
    print('done with work flow getting all rw-gen-sets')


def work_flow_with_gen_ranker():
    learn_all_rankers()
    get_all_reweighing_generated_sets()
    print('done with work flow within generating ranker files')

# TODO 将fav和unfav的参数，带上来，传进去
if __name__ == '__main__':
    # 先拿到训练好的模型 和原始的训练集作为输入
    # 1 根据train集的分布计算出四个权重值 in features.npy 2d-labels.npy
    # 2 将train集x 和y进行模型的
    work_flow_with_gen_ranker()
    # get_all_reweighing_generated_sets()
    print('done')

