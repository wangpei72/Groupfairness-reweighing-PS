import math
import sys

sys.path.append("../")
import numpy as np
import common_utils as cu


# 需要两类ranker，实际上是四个，分别是
# DP-ranker(up-fav) FP-ranker(p-fav)这两个按照proba升序排列，排在前面的，prob最小
# DN-ranker(up-unfav) FN-ranker(p-unfav)这两个按照proba降序排列，排在最前面的prob数值更大
# 是D还是F由sex是0还是1决定
# 是P还是N由y-label是1还是0决定

def get_conditionings_from_npy(x, y):
    x_origin = np.load(x)
    y_origin = np.load(y)

    shape_x = x_origin.shape
    shape_y = y_origin.shape
    protected_attribute_names = ['sex']
    protected_attributes = x_origin[:, 8]
    protected_attributes = protected_attributes[:, np.newaxis]
    labels = y_origin[:, 1]
    labels = labels[:, np.newaxis]

    (priv_cond, unpriv_cond, fav_cond, unfav_cond,
     cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) = \
        cu._obtain_conditionings(condition_dict_priv=[{'sex': 1}],
                                 condition_dict_unpriv=[{'sex': 0}],
                                 protected_attributes=protected_attributes,
                                 protected_attribute_names=protected_attribute_names,
                                 labels=labels,
                                 favorable_label=1.0,
                                 unfavorable_label=0.0)

    return cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav


def gen_all_sets(ranker_file, x, y):
    x_origin = np.load(x)
    y_origin = np.load(y)
    ranker_origin = np.load(ranker_file)
    x_up_fav, x_p_fav, x_up_unfav, x_p_unfav = \
        get_x_index_slices(x, y)
    DP_up_fav_asc, FP_p_fav_asc, DN_up_unfav_desc, FN_p_unfav_desc = \
        get_sorted_rankers(ranker_file, x, y)
    # DP FP DN FN
    wght_up_fav, wght_p_fav, wght_up_unfav, wght_p_unfav = get_weights_tuple(x_origin, y_origin)
    DP_set, DP_label = gen_DP_set(DP_up_fav_asc, x_up_fav, x_origin, y_origin, wght_up_fav)
    FP_set, FP_label = gen_FP_set(FP_p_fav_asc, x_p_fav, x_origin, y_origin, wght_p_fav)
    DN_set, DN_label = gen_DN_set(DN_up_unfav_desc, x_up_unfav, x_origin, y_origin, wght_up_unfav)
    FN_set, FN_label = gen_FN_set(FN_p_unfav_desc, x_p_unfav, x_origin, y_origin, wght_p_unfav)

    final_gen_set = np.concatenate((DP_set, FP_set, DN_set, FN_set), axis=0)
    final_gen_label = np.concatenate((DP_label, FP_label, DN_label, FN_label), axis=0)

    # 32561 -> 31380
    np.save('result_dataset/x_generated.npy', final_gen_set)
    np.save('result_dataset/y_generated.npy', final_gen_label)
    return final_gen_set, final_gen_label


def get_x_index_slices(x, y):
    cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav = \
        get_conditionings_from_npy(x, y)
    x_p_fav = np.argwhere(cond_p_fav)
    x_p_unfav = np.argwhere(cond_p_unfav)
    x_up_fav = np.argwhere(cond_up_fav)
    x_up_unfav = np.argwhere(cond_up_unfav)

    return x_up_fav, x_p_fav, x_up_unfav, x_p_unfav


def get_sorted_rankers(ranker_origin_npy, x, y):
    x_origin = np.load(x)
    y_origin = np.load(y)
    ranker_origin = np.load(ranker_origin_npy)
    shape_x = x_origin.shape

    protected_attributes = x_origin[:, 8]
    protected_attributes = protected_attributes[:, np.newaxis]
    labels = y_origin[:, 1]
    labels = labels[:, np.newaxis]

    instance_weights = np.ones((shape_x[0],), dtype=np.float64)
    n = np.sum(instance_weights, dtype=np.float64)

    #     我希望能返回四个np.ndarray, 分别包含着index的列表，index以原先的xy的index为准
    #     首先是按照四个condition来分成四个array，代表四大类，此时还没有排序
    cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav = \
        get_conditionings_from_npy(x, y)
    x_p_fav = np.argwhere(cond_p_fav)
    x_p_unfav = np.argwhere(cond_p_unfav)
    x_up_fav = np.argwhere(cond_up_fav)
    x_up_unfav = np.argwhere(cond_up_unfav)

    assert np.abs((x_p_fav.shape[0] + x_p_unfav.shape[0] + \
                   x_up_fav.shape[0] + x_up_unfav.shape[0]) - 32561) < 1e-6

    # 现在按照prob排序，排序先看prob（ascend or descend），再看index（ascend）
    # 我们有这四类的prob，现在思考拆分一下，如果要排列DP/(FP)的集合
    # 首先取x_up_fav,这个array包含了属于该集合的x_origin的index，拿这个index取prob，取到一个例如a
    # 查询prob值，然后a插入到新的list，继续遍历x_up_fav中的index，查prob，拿到b，按照DP的升序原则
    # prob越小越排在前面，如果这个prob b，小于a，b放在list中a的前面，
    # 若大于a，b放在list中a的后面，
    # 如果相同的话，按照索引顺序直接放在a后面,注意，实际上插入的元素都是x_origin的index
    # def insert_sort(l):
    #     length = len(l)
    #     for i in range(1, length):
    #         j = i - 1
    #
    #         while l[j] > l[i] and j >= 0:
    #             j -= 1
    #
    #         l.insert(j + 1, l.pop(i))
    #
    #     return l
    DP_up_fav_asc = sort_positive_ascend(x_up_fav, ranker_origin)
    FP_p_fav_asc = sort_positive_ascend(x_p_fav, ranker_origin)
    DN_up_unfav_desc = sort_negative_descend(x_up_unfav, ranker_origin)
    FN_p_unfav_desc = sort_negative_descend(x_p_unfav, ranker_origin)

    return DP_up_fav_asc, FP_p_fav_asc, DN_up_unfav_desc, FN_p_unfav_desc


def sort_positive_ascend(x_upri_or_pri, ranker_origin):
    P_upri_or_pri_ranker = []
    len = x_upri_or_pri.shape[0]
    # 初始化，将第一个放在待排序的list当中
    P_upri_or_pri_ranker.append(x_upri_or_pri[0, 0])
    for i in range(1, len):
        # 排序对象为index数值 cur_idx i是被操作对象
        cur_idx = x_upri_or_pri[i, 0]
        probi = ranker_origin[cur_idx, 1]
        j = i - 1
        before_idx = P_upri_or_pri_ranker[j]
        probj = ranker_origin[before_idx, 1]

        while probj > probi and j >= 0:
            j -= 1
            before_idx = P_upri_or_pri_ranker[j]
            probj = ranker_origin[before_idx, 1]
        P_upri_or_pri_ranker.insert(j + 1, cur_idx)

    # if True:
    #     print("++++++++++++++下面是验证，按照预期的，对于正样本prob应当由小到大排列+++++++++++")
    #     for item in P_upri_or_pri_ranker:
    #         print(ranker_origin[item, 1])
    return P_upri_or_pri_ranker


def sort_negative_descend(x_upri_or_pri, ranker_origin):
    N_upri_or_pri_ranker = []
    len = x_upri_or_pri.shape[0]
    # 初始化，将第一个放在待排序的list当中
    N_upri_or_pri_ranker.append(x_upri_or_pri[0, 0])
    for i in range(1, len):
        # 排序对象为index数值 cur_idx i是被操作对象
        cur_idx = x_upri_or_pri[i, 0]
        probi = - ranker_origin[cur_idx, 1]
        j = i - 1
        before_idx = N_upri_or_pri_ranker[j]
        probj = - ranker_origin[before_idx, 1]

        while probj > probi and j >= 0:
            j -= 1
            before_idx = N_upri_or_pri_ranker[j]
            probj = - ranker_origin[before_idx, 1]
        N_upri_or_pri_ranker.insert(j + 1, cur_idx)

    # if False:
    #     print("---------------下面是验证，按照预期的，对于负样本prob应当由大到小-------------")
    #     for item in N_upri_or_pri_ranker:
    #         print(ranker_origin[item, 1])
    return N_upri_or_pri_ranker


def get_weights_tuple(x_origin, y_origin):
    shape_x = x_origin.shape
    shape_y = y_origin.shape
    protected_attribute_names = ['sex']
    protected_attributes = x_origin[:, 8]
    protected_attributes = protected_attributes[:, np.newaxis]
    labels = y_origin[:, 1]
    labels = labels[:, np.newaxis]

    instance_weights = np.ones((shape_x[0],), dtype=np.float64)
    (priv_cond, unpriv_cond, fav_cond, unfav_cond,
     cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) = \
        cu._obtain_conditionings(condition_dict_priv=[{'sex': 1}],
                                 condition_dict_unpriv=[{'sex': 0}],
                                 protected_attributes=protected_attributes,
                                 protected_attribute_names=protected_attribute_names,
                                 labels=labels,
                                 favorable_label=1.0,
                                 unfavorable_label=0.0)
    weights_dict = cu.fit(instance_weights,
                          priv_cond, unpriv_cond, fav_cond, unfav_cond,
                          cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav
                          )
    # DP FP DN FN
    return weights_dict['WUPF'], weights_dict['WPF'], weights_dict['WUPUF'], weights_dict['WPUF']

# 8： 将2份DP（劣势正样本）的复制增加到D_ps，
# 9：增加（2.19 - 2*|DP|）的向下取整结果 个lowest-排序的（从bottom往top取）元素的DP（劣正）到D_ps
# 10: 增加 （0.85*|DN|）的向下取整结果 个lowest-ranked 元素的DN（劣负）到D_ps(==换言之删除掉最靠近boundary（负样本概率较大的那边)的0.15%|DN|==）
#
# 11: 增加（0.79*|FP|）的向下取整结果个 highest-ranked 元素的FP（优正）到D_ps (==换言之删除掉最靠近boundary（负样本概率较大的那边)的0.21%|FP|==）
#
# 12：增加 1 份FN（优负）的复制到D_ps
#
# 13：增加（1.09 - 1）*|FN|的向下取整结果个 highest-ranked 元素的FN（优负）增加到D_ps(==换言之循环置底的方式重复最靠近boundary（正样本概率较低的那边)的0.15%|DN|==）
def gen_DP_set(DP_up_fav_asc, x_up_fav, x_origin, y_origin, wght_up_fav):
    wght_floor = math.floor(wght_up_fav)
    wght_remainder = wght_up_fav - wght_floor
    # instance_weights = np.ones((x_origin.shape[0],), dtype=np.float64)
    DP_num = x_up_fav.shape[0]
    remainder_num = np.floor(DP_num * wght_remainder)
    cell_x = x_origin[x_up_fav[:, 0], :]
    cell_y = y_origin[x_up_fav[:, 0], :]
    DP_set = cell_x
    DP_label = cell_y
    if wght_floor >= 2:
        for i in range(wght_floor - 1):
            DP_set = np.concatenate((DP_set, cell_x), axis=0)
            DP_label = np.concatenate((DP_label, cell_y), axis=0)
    tmp = 0
    for j in range(int(remainder_num)):
        #         找排在最前面的 循环置底
        DP_set = np.concatenate((DP_set, x_origin[DP_up_fav_asc[tmp], np.newaxis]), axis=0)
        DP_label = np.concatenate((DP_label, y_origin[DP_up_fav_asc[tmp], np.newaxis]), axis=0)
        tmp += 1
    return DP_set, DP_label


def gen_FN_set(FN_p_unfav_desc, x_p_unfav, x_origin, y_origin, wght_p_unfav):
    wght_floor = math.floor(wght_p_unfav)
    wght_remainder = wght_p_unfav - wght_floor
    # instance_weights = np.ones((x_origin.shape[0],), dtype=np.float64)
    FN_num = x_p_unfav.shape[0]
    remainder_num = np.floor(FN_num * wght_remainder)
    cell_x = x_origin[x_p_unfav[:, 0], :]
    cell_y = y_origin[x_p_unfav[:, 0], :]
    FN_set = cell_x
    FN_label = cell_y
    if wght_floor > 2:
        for i in range(wght_floor - 1):
            FN_set = np.concatenate((FN_set, cell_x), axis=0)
            FN_label = np.concatenate((FN_label, cell_y), axis=0)
    tmp = 0
    for j in range(int(remainder_num)):
        #         找排在最前面的 循环置底
        FN_set = np.concatenate((FN_set, x_origin[FN_p_unfav_desc[tmp], np.newaxis]), axis=0)
        FN_label = np.concatenate((FN_label, y_origin[FN_p_unfav_desc[tmp], np.newaxis]), axis=0)
        tmp += 1
    return FN_set, FN_label


def gen_DN_set(DN_up_unfav_desc, x_up_unfav, x_origin, y_origin, wght_up_unfav):
    global DN_set, DN_label
    wght_floor = math.floor(wght_up_unfav)  # 0 0.85
    wght_remainder = wght_up_unfav - wght_floor
    # instance_weights = np.ones((x_origin.shape[0],), dtype=np.float64)
    DN_num = x_up_unfav.shape[0]
    remainder_num = np.floor(DN_num * wght_remainder)
    tmp = DN_num - 1

    for j in range(int(remainder_num)):
        #         找排在最前面的 循环置底
        if j == 0:
            DN_set = x_origin[DN_up_unfav_desc[tmp], np.newaxis]
            DN_label = y_origin[DN_up_unfav_desc[tmp], np.newaxis]
            tmp -= 1
        else:
            DN_set = np.concatenate((DN_set, x_origin[DN_up_unfav_desc[tmp], np.newaxis]), axis=0)
            DN_label = np.concatenate((DN_label, y_origin[DN_up_unfav_desc[tmp], np.newaxis]), axis=0)
            tmp -= 1
    return DN_set, DN_label


def gen_FP_set(FP_p_fav_asc, x_p_fav, x_origin, y_origin, wght_p_fav):
    global FP_set, FP_label
    wght_floor = math.floor(wght_p_fav)  # 0 0.85
    wght_remainder = wght_p_fav - wght_floor
    # instance_weights = np.ones((x_origin.shape[0],), dtype=np.float64)
    FP_num = x_p_fav.shape[0]
    remainder_num = np.floor(FP_num * wght_remainder)
    tmp = FP_num - 1

    for j in range(int(remainder_num)):
        #         找排在最前面的 循环置底
        if j == 0:
            FP_set = x_origin[FP_p_fav_asc[tmp], np.newaxis]
            FP_label = y_origin[FP_p_fav_asc[tmp], np.newaxis]
            tmp -= 1
        else:
            FP_set = np.concatenate((FP_set, x_origin[FP_p_fav_asc[tmp], np.newaxis]), axis=0)
            FP_label = np.concatenate((FP_label, y_origin[FP_p_fav_asc[tmp], np.newaxis]), axis=0)
            tmp -= 1
    return FP_set, FP_label


if __name__ == '__main__':
    # prob_origin = np.load('ranker_result_origin/2dims_result.npy')
    # x_origin = np.load('../data/adult/data-x.npy')
    # y_origin = np.load('../data/adult/data-y.npy')
    # get_sorted_rankers('ranker_result_origin/2dims_result.npy',
    #                    '../data/adult/data-x.npy',
    #                    '../data/adult/data-y.npy')
    gen_all_sets('ranker_result_origin/2dims_result.npy',
                       '../data/adult/data-x.npy',
                       '../data/adult/data-y.npy')
    print('end')
