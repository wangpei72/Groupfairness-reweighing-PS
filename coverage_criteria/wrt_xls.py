import xlwt as wt
import numpy as np
import time
import sys
sys.path.append("../")


def wrt_xls_file(output_filename, sheet_name, result_name, path_prefix ='../adult-res-nc', id_list_cnt = 0):
    id_list = ['01', '02', '03', '04', '05']
    target_arr = np.load(path_prefix + '/20_tests_' + id_list[id_list_cnt] + '.npy', allow_pickle=True)
    shape = target_arr.shape
    idx = 0
    if(len(shape) == 1):
        workbook = wt.Workbook()
        while idx < 5:
            target_arr = np.load(path_prefix + '/20_tests_' + id_list[idx] + '.npy', allow_pickle=True)
            print("arr shape is %d" % target_arr.shape)
            sheet = workbook.add_sheet(sheet_name + id_list[idx])
            sheet.write(0, 0, 'test id')
            sheet.write(1, 0, result_name)
            for i in range(shape[0]):
                sheet.write(0, i + 1, i + 1)
                sheet.write(1, i + 1, str(target_arr[i]))
            idx += 1
        workbook.save(path_prefix + '/xls_file' + output_filename)
        print('test res saved as xls file.')


if __name__ == "__main__":
    # acc_array = np.load('./test_accu20220220145027.npy', allow_pickle=True)
    id_list = ['avg', '02', '03', '04', '05']
    id_list_cnt = 0

    workbook = wt.Workbook()
    while id_list_cnt < 1:
        eoop_array = np.load('bank-testres/res_avg/eoop_avg.npy')
        eood_array = np.load('bank-testres/res_avg/eood_avg.npy')
        acc_arry = np.load('bank-testres/res_avg/test_accu_avg.npy', allow_pickle=True)
        # di_array = np.load('bank-adult-testres/di_res' + id_list[id_list_cnt] + '.npy')
        # spd_array = np.load('bank-adult-testres/spd_res' + id_list[id_list_cnt] + '.npy')
        # eoop_array = np.load('bank-adult-testres/eoop_res' + id_list[id_list_cnt] + '.npy')
        # eood_array = np.load('bank-adult-testres/eood_res' + id_list[id_list_cnt] + '.npy')

        sheet = workbook.add_sheet('eoop_and_eood' + id_list[id_list_cnt])
        sheet.write(0, 0, 'test id')
        sheet.write(1, 0, 'equality_of_oppo')
        sheet.write(2, 0, 'equality_of_odds')
        sheet.write(3, 0, 'predict_accuracy')
        for i in range(20):
            sheet.write(0, i + 1, i + 1)
            sheet.write(1, i + 1, str(eoop_array[i]))
            sheet.write(2, i + 1, str(eood_array[i]))
            sheet.write(3, i + 1, str(acc_arry[i]))
        id_list_cnt += 1
    workbook.save('./bank-adult-testres/xls_file/bank_eoop_eood_avg_results.xls')
    print('test res saved as xls file.')
