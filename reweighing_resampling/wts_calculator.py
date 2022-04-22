import sys
sys.path.append("../")
import numpy as np
import common_utils as cu




if __name__ == '__main__':
    # 先获取原始数据
    x_origin = np.load('../data/adult/data-x.npy')
    y_origin = np.load('../data/adult/data-y.npy')

    # 优势劣势群体、敏感属性的定义
#     包含在fit和transfrom中
    shape_x = x_origin.shape
    shape_y = y_origin.shape
    protected_attribute_names = ['sex']
    protected_attributes = x_origin[:, 8]
    protected_attributes = protected_attributes[:, np.newaxis]
    labels = y_origin[:, 1]
    labels = labels[:, np.newaxis]

    instance_weights = np.ones((shape_x[0],), dtype=np.float64)
    (priv_cond, unpriv_cond, fav_cond, unfav_cond,
     cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) =\
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
    print(weights_dict)
    instance_weights = cu.transform(instance_weights, cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav,
                                    weights_dict['WPF'], weights_dict['WPUF'],
                                    weights_dict['WUPF'], weights_dict['WUPUF'])



