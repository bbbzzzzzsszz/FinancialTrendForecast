#-*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
from sklearn import preprocessing
import os


def process_base_company(company_data, test_time):
    NAPS = []
    ROE = []
    ROAEBIT = []
    ROIC = []
    Totprfcostrt = []
    Opeprfrt = []
    Reporttype = []

    for k, v in company_data.iterrows():
        Comcd_val = v['上市公司代码_Comcd']
        Lcomnm_val = v['最新公司全称_Lcomnm']
        Enddt_val = v['截止日期_Enddt']
        Reporttype_val = v['报表类型_Reporttype']

        NAPS_val = v['每股净资产(元/股)_NAPS']
        ROE_val = v['净资产收益率(摊薄)(%)_ROE']
        ROAEBIT_val = v['资产报酬率(%)_ROAEBIT']
        ROIC_val = v['投入资本回报率(%)_ROIC']
        Totprfcostrt_val = v['成本费用利润率(%)_Totprfcostrt']
        Opeprfrt_val = v['营业利润率(%)_Opeprfrt']

        NAPS.append(NAPS_val)
        ROE.append(ROE_val)
        ROAEBIT.append(ROAEBIT_val)
        ROIC.append(ROIC_val)
        Totprfcostrt.append(Totprfcostrt_val)
        Opeprfrt.append(Opeprfrt_val)
        Reporttype.append(int(Reporttype_val[-1]))

    train_list = []
    label_list = []
    if test_time == "Q1":
        nums = 16
    elif test_time == "Q2":
        nums = 17
    elif test_time == "Q3":
        nums = 18
    elif test_time == "Q4":
        nums = 19
    else:
        print("Please input Q1~Q4")
        os._exit(0)
    for i in range(nums, nums+1):
        temp_list = []
        temp_list.append(Reporttype[i+8])
        #当前节点与之前节点的差值
        temp_list.append(NAPS[i+7]-NAPS[i+6])
        temp_list.append(ROE[i+7]-ROE[i+6])
        temp_list.append(ROAEBIT[i+7]-ROAEBIT[i+6])
        temp_list.append(ROIC[i+7]-ROIC[i+6])
        temp_list.append(Totprfcostrt[i+7]-Totprfcostrt[i+6])
        temp_list.append(Opeprfrt[i+7]-Opeprfrt[i+6])
        # 各种指标近2年的均值
        # +1*6
        temp_list.append(sum(NAPS[i: i+8])/8.0)
        temp_list.append(sum(ROE[i: i+8])/8.0)
        temp_list.append(sum(ROAEBIT[i: i+8])/8.0)
        temp_list.append(sum(ROIC[i: i+8])/8.0)
        temp_list.append(sum(Totprfcostrt[i: i+8])/8.0)
        temp_list.append(sum(Opeprfrt[i: i+8])/8.0)

        # 各种指标1年的均值
        # +1*6
        temp_list.append(sum(NAPS[i+4: i+8])/8.0)
        temp_list.append(sum(ROE[i+4: i+8])/8.0)
        temp_list.append(sum(ROAEBIT[i+4: i+8])/8.0)
        temp_list.append(sum(ROIC[i+4: i+8])/8.0)
        temp_list.append(sum(Totprfcostrt[i+4: i+8])/8.0)
        temp_list.append(sum(Opeprfrt[i+4: i+8])/8.0)

        # 近一年的各季度相对于前两年各季度 每个指标的增益情况 
        # +4*6
        for t in range(4):
            temp_list.append(NAPS[i+t+4] - NAPS[i+t])
        for t in range(4):
            temp_list.append(ROE[i+t+4] - ROE[i+t])
        for t in range(4):
            temp_list.append(ROAEBIT[i+t+4] - ROAEBIT[i+t])
        for t in range(4):
            temp_list.append(ROIC[i+t+4] - ROIC[i+t])
        for t in range(4):
            temp_list.append(Totprfcostrt[i+t+4] - Totprfcostrt[i+t])
        for t in range(4):
            temp_list.append(Opeprfrt[i+t+4] - Opeprfrt[i+t])

        # 近2年各指标为正的次数，为负的次数
        # +2 * 6
        temp_list.append(sum([1 for i in NAPS[i: i+8] if i>0]))
        temp_list.append(sum([1 for i in ROE[i: i+8] if i>0]))
        temp_list.append(sum([1 for i in ROAEBIT[i: i+8] if i>0]))
        temp_list.append(sum([1 for i in ROIC[i: i+8] if i>0]))
        temp_list.append(sum([1 for i in Totprfcostrt[i: i+8] if i>0]))
        temp_list.append(sum([1 for i in Opeprfrt[i: i+8] if i>0]))

        temp_list.append(sum([1 for i in NAPS[i: i+8] if i<0]))
        temp_list.append(sum([1 for i in ROE[i: i+8] if i<0]))
        temp_list.append(sum([1 for i in ROAEBIT[i: i+8] if i<0]))
        temp_list.append(sum([1 for i in ROIC[i: i+8] if i<0]))
        temp_list.append(sum([1 for i in Totprfcostrt[i: i+8] if i<0]))
        temp_list.append(sum([1 for i in Opeprfrt[i: i+8] if i<0]))

        # new add 
        # 各指标近1年的均值和各指标近2年均值的差值
        # +1*6
        temp_list.append( (sum(NAPS[i+4: i+8]) - sum(NAPS[i: i+4])) / 4.0 )
        temp_list.append( (sum(ROE[i+4: i+8]) - sum(ROE[i: i+4])) / 4.0 )
        temp_list.append( (sum(ROAEBIT[i+4: i+8]) - sum(ROAEBIT[i: i+4])) / 4.0 )
        temp_list.append( (sum(ROIC[i+4: i+8]) - sum(ROIC[i: i+4])) / 4.0 )
        temp_list.append( (sum(Totprfcostrt[i+4: i+8]) - sum(Totprfcostrt[i: i+4])) / 4.0 )
        temp_list.append( (sum(Opeprfrt[i+4: i+8]) - sum(Opeprfrt[i: i+4])) / 4.0 )

        # 去年该季度是否上升
        temp_list.append(1 if (sum(Opeprfrt[i-4: i+4])/4.0> Opeprfrt[i+4]) else 0)
        
        # 2年前该季度是否上升
        temp_list.append(1 if (sum(Opeprfrt[i-8: i])/4.0> Opeprfrt[i]) else 0)

        label = 1 if (sum(Opeprfrt[i: i+8])/8.0 -  Opeprfrt[i+8]>0) else 0

        train_list.append(temp_list)
        label_list.append(label)
    
    return train_list, label_list


def read_data_test(csv_file, test_time):
    data = pd.read_csv(csv_file, encoding="gbk", header=0)

    company_id = set(data['上市公司代码_Comcd'].tolist())
    train_all = []
    label_all = []
    for cid in company_id:
        # print(i)
        company_data = data[data['上市公司代码_Comcd'] == cid]
        company_data = company_data.sort_values(by = '截止日期_Enddt', ascending=True)  # , inplace=True
        company_data.reset_index(inplace=True)
        train_list, label_list = process_base_company(company_data, test_time)
        train_all += train_list
        label_all += label_list
 
    return train_all, label_all


def test(csv_file, model_save_path, test_time):
    # load train model
    clf_load = joblib.load(model_save_path)   # 'model.pkl

    test_all, label_all =  read_data_test(csv_file, test_time)
    print("Test sample nums: ", len(test_all))
    print('Rise nums: ', sum(label_all))

    delete_col = [46, 59, 24,28,61,4,32]
    test_X = np.array(test_all).reshape(len(test_all), -1)
    test_X = np.delete(test_X, delete_col, axis=1)
    
    test_X = preprocessing.normalize(test_X, norm='l2')
    
    test_Y = np.array(label_all)

    clf_load = joblib.load(model_save_path)   # 'model.pkl
    test_pre = clf_load.predict(test_X, )
    print("Accuracy in test file: {}".format(metrics.accuracy_score(test_Y, test_pre)))


if __name__ == "__main__":
    csv_file = "data/data2012.csv"
    model_save_path = "model/model.pkl"
    test_time = "Q1"    # Q1  or  Q2  or Q3   or Q4
    test(csv_file, model_save_path, test_time)

    
