from .utils import compute_boolean_conditioning_vector, compute_num_instances, compute_num_TF_PN, compute_num_gen_TF_PN
import numpy as np

bank_map = ['age', 'job', 'marital', 'education', 'education-num', 'marital-status',
               'occupation', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'xx', 'xxx',
            'xxxx', 'xxxxx', 'yy', 'yyyy', 'yyyyy']
bank_protect_attr = ['age']
bank_favor_map = [{'yes': 1}, {'no': 0}]

default_map = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

"""
return difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                      priv_group=priv_group, pos_label=pos_label,
                      sample_weight=sample_weight)
"""
def E_Oppo(X, y_true, y_pre, feature_names=None):
    if feature_names is None:
        feature_names = default_map
    # 越接近0越公平
    unprivileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[{'sex': 0}])
    privileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[{'sex': 1}])
    unpri_instances_num = np.sum(unprivileged_boolean_condition_vector != 0, dtype=np.float32)
    pri_instances_num = np.sum(privileged_boolean_condition_vector != 0, dtype=np.float32)

    w = np.ones(y_pre.shape)
    unpri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'sex': 0}])
    pri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'sex': 1}])

    TPR_diff = unpri_dict['TP'] / unpri_instances_num - pri_dict['TP'] / pri_instances_num
    FPR_diff = unpri_dict['FP'] / unpri_instances_num - pri_dict['FP'] / pri_instances_num

    return TPR_diff


def E_Oppo_adult_age(X, y_true, y_pre, feature_names=None):
    if feature_names is None:
        feature_names = default_map
    # 越接近0越公平
    unprivileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[{'age': 1}, {'age': 2}, {'age': 3}])
    privileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[ {'age': 4}, {'age': 5},
                                                                                            {'age': 6}, {'age': 7}, {'age': 8},
                                                                                            {'age': 9}])
    unpri_instances_num = np.sum(unprivileged_boolean_condition_vector != 0, dtype=np.float32)
    pri_instances_num = np.sum(privileged_boolean_condition_vector != 0, dtype=np.float32)

    w = np.ones(y_pre.shape)
    unpri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'age': 1}, {'age': 2}, {'age': 3}])
    pri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[ {'age': 4}, {'age': 5},
                                                                                            {'age': 6}, {'age': 7}, {'age': 8},
                                                                                            {'age': 9}])

    TPR_diff = unpri_dict['TP'] / unpri_instances_num - pri_dict['TP'] / pri_instances_num
    FPR_diff = unpri_dict['FP'] / unpri_instances_num - pri_dict['FP'] / pri_instances_num

    return TPR_diff


def E_Odds_adult_age(X, y_true, y_pre, feature_names=None):
    if feature_names is None:
        feature_names = default_map
    # 越接近0越公平
    unprivileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[{'age': 1}, {'age': 2}, {'age': 3}])
    privileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[ {'age': 4}, {'age': 5},
                                                                                            {'age': 6}, {'age': 7}, {'age': 8},
                                                                                            {'age': 9}])
    unpri_instances_num = np.sum(unprivileged_boolean_condition_vector != 0, dtype=np.float32)
    pri_instances_num = np.sum(privileged_boolean_condition_vector != 0, dtype=np.float32)

    w = np.ones(y_pre.shape)
    unpri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'age': 1}, {'age': 2}, {'age': 3}])
    pri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[ {'age': 4}, {'age': 5},
                                                                                            {'age': 6}, {'age': 7}, {'age': 8},
                                                                                            {'age': 9}])

    TPR_diff = unpri_dict['TP'] / unpri_instances_num - pri_dict['TP'] / pri_instances_num
    FPR_diff = unpri_dict['FP'] / unpri_instances_num - pri_dict['FP'] / pri_instances_num

    return (TPR_diff + FPR_diff) / 2


def E_Odds(X, y_true, y_pre, feature_names=None):
    if feature_names is None:
        feature_names = default_map
    # 越接近0越公平
    unprivileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[{'sex': 0}])
    privileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[{'sex': 1}])
    unpri_instances_num = np.sum(unprivileged_boolean_condition_vector != 0, dtype=np.float32)
    pri_instances_num = np.sum(privileged_boolean_condition_vector != 0, dtype=np.float32)

    w = np.ones(y_pre.shape)
    unpri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'sex': 0}])
    pri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'sex': 1}])

    TPR_diff = unpri_dict['TP'] / unpri_instances_num - pri_dict['TP'] / pri_instances_num
    FPR_diff = unpri_dict['FP'] / unpri_instances_num - pri_dict['FP'] / pri_instances_num

    return (TPR_diff + FPR_diff) / 2


def E_Oppo_bank(X, y_true, y_pre, feature_names=None):
    if feature_names is None:
        feature_names = bank_map
    # 越接近0越公平
    unprivileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[{'age': 1}, {'age': 2}, {'age': 3}])
    privileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names, condition=[ {'age': 4}, {'age': 5},
                                                                                            {'age': 6}, {'age': 7}, {'age': 8},
                                                                                            {'age': 9}])
    unpri_instances_num = np.sum(unprivileged_boolean_condition_vector != 0, dtype=np.float32)
    pri_instances_num = np.sum(privileged_boolean_condition_vector != 0, dtype=np.float32)

    w = np.ones(y_pre.shape)
    unpri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'age': 1}, {'age': 2}, {'age': 3}])
    pri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'age': 4}, {'age': 5},
                                                                                    {'age': 6}, {'age': 7}, {'age': 8},
                                                                                    {'age': 9}])

    TPR_diff = unpri_dict['TP'] / unpri_instances_num - pri_dict['TP'] / pri_instances_num
    FPR_diff = unpri_dict['FP'] / unpri_instances_num - pri_dict['FP'] / pri_instances_num

    return TPR_diff


def E_Odds_bank(X, y_true, y_pre, feature_names=None):
    if feature_names is None:
        feature_names = bank_map
    unprivileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names,
                                                                                condition=[{'age': 1}, {'age': 2},
                                                                                           {'age': 3}])
    privileged_boolean_condition_vector = compute_boolean_conditioning_vector(X, feature_names=feature_names,
                                                                              condition=[{'age': 4}, {'age': 5}, {'age': 6},
                                                                                        {'age': 7}, {'age': 8}, {'age': 9}])
    unpri_instances_num = np.sum(unprivileged_boolean_condition_vector != 0, dtype=np.float32)
    pri_instances_num = np.sum(privileged_boolean_condition_vector != 0, dtype=np.float32)
    w = np.ones(y_pre.shape)
    unpri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0.,
                                   condition=[{'age': 1}, {'age': 2}, {'age': 3}])
    pri_dict = compute_num_TF_PN(X, y_true, y_pre, w, bank_map, 1., 0., condition=[{'age': 4}, {'age': 5},
                                                                                   {'age': 6}, {'age': 7}, {'age': 8},
                                                                                   {'age': 9}])
    TPR_diff = unpri_dict['TP'] / unpri_instances_num - pri_dict['TP'] / pri_instances_num
    FPR_diff = unpri_dict['FP'] / unpri_instances_num - pri_dict['FP'] / pri_instances_num

    return (TPR_diff + FPR_diff) / 2
