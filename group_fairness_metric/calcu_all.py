import sys

sys.path.append("../")
from . import statistical_parity_difference, disparte_impact, equality_of_oppo


def calcu_all_metrics(X_arr, y_arr, y_true_arr):
    spd = statistical_parity_difference.S_P_D(X_arr, y_arr)
    di = disparte_impact.D_I(X_arr, y_arr)
    eoop = equality_of_oppo.E_Oppo(X_arr, y_true_arr, y_arr)
    eood = equality_of_oppo.E_Odds(X_arr, y_true_arr, y_arr)
    return spd, di, eoop, eood


if __name__ == '__main__':
    pass
