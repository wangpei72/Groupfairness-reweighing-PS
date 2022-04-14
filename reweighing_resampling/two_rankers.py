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


def get_sorted_ranker(ranker_origin_npy, x, y):
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
    cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav =\
    get_conditionings_from_npy(x, y)
    x_p_fav = np.argwhere(cond_p_fav)
    x_p_unfav = np.argwhere(cond_p_unfav)
    x_up_fav = np.argwhere(cond_up_fav)
    x_up_unfav = np.argwhere(cond_up_unfav)

    assert np.abs((x_p_fav.shape[0] + x_p_unfav.shape[0] +\
           x_up_fav.shape[0] + x_up_unfav.shape[0]) - 32561 )< 1e-6

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
    DP_up_fav_ranker = []
    len = x_up_fav.shape[0]
    # 初始化，将第一个放在待排序的list当中
    DP_up_fav_ranker.append(x_up_fav[0, 0])
    for i in range(1, len):
        # 排序对象为index数值 cur_idx i是被操作对象
        cur_idx = x_up_fav[i, 0]
        probi = ranker_origin[cur_idx, 1]
        # 上一个！发现错误，我要看的应该是排序当中的list的上一个
        j = i - 1
        before_idx = DP_up_fav_ranker[j]
        probj = ranker_origin[before_idx, 1]

        while probj > probi and j >= 0:
            j -= 1
            before_idx = DP_up_fav_ranker[j]
            probj = ranker_origin[before_idx, 1]
        DP_up_fav_ranker.insert(j + 1, cur_idx)

    print("下面是验证，按照预期的，prob应当由小到大排列")
    for item in DP_up_fav_ranker:
        print(ranker_origin[item, 1])

    return DP_up_fav_ranker


if __name__ == '__main__':
    # prob_origin = np.load('ranker_result_origin/2dims_result.npy')
    # x_origin = np.load('../data/census/data-x.npy')
    # y_origin = np.load('../data/census/data-y.npy')
    get_sorted_ranker('ranker_result_origin/2dims_result.npy',
                        '../data/census/data-x.npy',
                        '../data/census/data-y.npy')

    print('end')
