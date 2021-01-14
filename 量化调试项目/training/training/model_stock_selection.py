# coding: utf-8
import QUANTAXIS as QA
import pandas as pd
import tools
import time as tm
import training.config as cn
from pandas import to_datetime
import os
from datetime import datetime
import training.config as cn
from training import yinzi
import copy
import numpy as np
from training.model_train import  model_train
import model_config as mc

def close_down(data_copy):
    """
    最近三天的存在两天收盘价<0的情况
    :param data_copy
    :return: data_copy
    """
    factor1 = data_copy['收盘价涨跌幅'].tolist()
    factor1 = [1 if factor1[i] < 0 else 0 for i in range(len(factor1))]

    factor = []
    for i in range(len(factor1)):
        if i >= 3:
            b = sum(factor1[i - 3 + 1:i + 1])
            if b >= 2:
                l1 = 1
            else:
                l1 = 0
        else:
            l1 = 0
        factor.append(l1)
    data_copy['colse_down'] = factor
    return data_copy

def close_down2(data_copy):
    """
    最近三天的存在两天收盘价<0的情况
    :param data_copy
    :return: data_copy
    """
    factor1 = data_copy['收盘价涨跌幅'].tolist()
    factor1 = [1 if factor1[i] < 0 else 0 for i in range(len(factor1))]

    factor = []
    for i in range(len(factor1)):
        if i >= 6:
            b = sum(factor1[i - 6 + 1:i - 2])
            if b <=1:
                l1 = 1
            else:
                l1 = 0
        else:
            l1 = 0
        factor.append(l1)
    data_copy['colse_down2'] = factor
    return data_copy

def close_down3(data_copy):
    """
    最近三天的存在两天收盘价<0的情况
    :param data_copy
    :return: data_copy
    """
    #print(data_copy.columns)
    factor1 = data_copy['当日10点ma5'].tolist()

    factor = []
    for i in range(len(factor1)):
        if i >= 3:
            l1 = sum(factor1[i - 3 + 1:i +1])/len(factor1[i - 3 + 1:i +1])
        else:
            l1 = factor1[i]
        factor.append(l1)
    factor0 = (np.array(data_copy['当日10点价格'].tolist())/np.array(factor)-1)*100
    data_copy['colse_down3'] = factor0
    return data_copy

def factor_chuli3(data_pre,dir_yuce4):
    """
     对合并后的数据进行筛选
     :param data_pre: 完全合并后的预测明细
     :param ,dir_yuce4: 最终的需要的选股数据，筛选后的
     :return: 无
     """
    data_copy = copy.deepcopy(data_pre)
    df9 = set(data_copy[data_copy['factor1'] ==1].index.tolist())
    df91 = set(data_copy[data_copy['factor2'] ==1].index.tolist())
    df92 = set(data_copy[data_copy['factor3'] ==1].index.tolist())
    df93 = set(data_copy[data_copy['factor4'] ==1].index.tolist())
    df94 = set(data_copy[data_copy['factor5'] ==1].index.tolist())
    df95 = set(data_copy[data_copy['factor6'] ==1].index.tolist())
    df96 = set(data_copy[data_copy['factor7'] == 1].index.tolist())
    df97 = set(data_copy[data_copy['factor8'] == 1].index.tolist())

    df98 = set(data_copy[data_copy['收盘价涨跌幅'] <0].index.tolist())
    df99 = set([i + 1 for i in df98])
    df10 = set([i + 2 for i in df98])
    data_copy = close_down(data_copy)
    df11 = set(data_copy[data_copy['colse_down'] == 1].index.tolist())

    df12 = set(data_copy[data_copy['当日10点涨幅'] > 0].index.tolist())
    df13 = set(data_copy[data_copy['当日10点量比'] >1].index.tolist())
    df14 = set(data_copy[data_copy['日线收盘价/开盘价'] < 0].index.tolist())
    df15 = set(data_copy[data_copy['收盘价/开盘价'] > 1].index.tolist())

    data_copy = close_down2(data_copy)
    df16 = set(data_copy[data_copy['colse_down2'] == 1].index.tolist())
    df17 = set(data_copy[data_copy['colse_down3'] <0].index.tolist())
    df18 = set(data_copy[data_copy['收盘价涨跌幅'] > -8].index.tolist())
    df19 = set(data_copy[data_copy['涨停2'] < 9].index.tolist())

    df7 =  df9&df94 & df96&df11&df12&df13&df15&df16&df17&df18&df19 #& (df98 | df99)&(df98 | df10)

    df7 = list(df7)
    data_pre = data_pre.ix[df7, :]
    data_pre.to_csv(dir_yuce4, mode='w', index=False, header=True)

def factor_chuli2(dir_yuce2,save_yinzi,dir_yuce3):
    """
    对因子进行二次处理，并进行合并
    :param dir_yuce2: 初步合并后的预测明细
    :param save_yinzi: 需要二次处理的因子的数据
    :param dir_yuce3: 二次合并后的预测明细
    :return:
    """
    data1 = pd.read_table(dir_yuce2, sep=',')  # 读取训练数据集
    data2 = pd.read_table(save_yinzi, sep=',')  # 读取训练数据集
    print(len(data1), len(data2))
    data_copy = copy.deepcopy(data2)
    data_copy = yinzi.Factor1(data_copy,0,10,5)
    data_copy = yinzi.Factor2(data_copy, -0.1, 1, 10, 7)
    data_copy = yinzi.Factor3(data_copy, -0.1, 0.8, 10, 7)
    data_copy = yinzi.Factor4(data_copy, 0, 10, 6)
    data_copy = yinzi.Factor5(data_copy, -3, 3, 10, 6)
    data_copy = yinzi.Factor6(data_copy, 0.15)
    data_copy = yinzi.Factor7(data_copy, 0.1)
    data_copy = yinzi.Factor8(data_copy, 2, 10, 1)

    data1 = close_down3(data1)


    data2 = data_copy[['code', 'date','factor1','factor2','factor3',
                   'factor4','factor5','factor6','factor7','factor8']]
    data2.columns = ['code', 'date','factor1','factor2','factor3',
                   'factor4','factor5','factor6','factor7','factor8']
    print(len(data1),len(data2))
    result = pd.merge(data1, data2, on=['code', 'date'])
    print(len(result))
    result.to_csv(dir_yuce3, mode='w', index=False, header=True)

def factor_chuli0(save_yinzi2,save_yinzi):
    """
     生成需要合并的因子数据，保存为一个文件
     :param save_yinzi: 合并时需要的因子文件，这个因子文件时固定的，不需要二次处理
     :return:
     """

    shuju_file = cn.huice_shujuji_dir  # 读取数据集的目录
    data = pd.read_table(shuju_file, sep=',')  # 读取训练数据集
    data.rename(columns={'daima': 'code', 'data': 'date'}, inplace=True)

    data2 = data[['code', 'date','2','112','120','31','32','34','35', '136', '137', '138', '139', '140', '141','142' ,'涨停1', '涨停2', '涨停3', '涨停4', '涨停5']]
    data2.columns = ['code', 'date', '收盘价涨跌幅','当日10点涨幅', '当日10点量比', '日线收盘价/开盘价','收盘价/开盘价', '最低价/收盘价','最高价/开盘价', '当日10点ma5', '当日10点半ma5', '昨日10点ma5', '昨日10点半ma5', '当日10点价格', '当日10点半价格', '换手率', '涨停1',
                     '涨停2', '涨停3', '涨停4', '涨停5']
    #'factor1','factor2','factor3','factor4','factor5','factor6','factor7','factor8'
    data2.to_csv(save_yinzi2, mode='w', index=False, header=True)

    data3 = data[['code', 'date', '128', '129', '130', '131', '132', '133', '134', '135']]
    data3.to_csv(save_yinzi, mode='w', index=False, header=True)

def factor_chuli1(dir_yuce1,dir_yuce2,save_yinzi2,save_yinzi):
    """
     对预测明细的股票与各个因子进行合并
     :param dir_yuce1: 初始的预测明细
     :param ,dir_yuce2,save_yinzi: 初步合并后的预测明细，与
     合并时需要的因子文件，这个因子文件时固定的，不需要二次处理
     :return:
     """
    factor_chuli0(save_yinzi2,save_yinzi)
    data1 = pd.read_table(dir_yuce1, sep=',', header=None)  # 读取训练数据集
    data2 = pd.read_table(save_yinzi2, sep=',')  # 读取训练数据集
    data1.columns = ['code', 'date', 'rise_down', 'class', 'pro', 'pro1', 'pro2', 'pro3']
    data11 = to_datetime(data1['date'], format="%Y/%m/%d")
    data1['date'] = [datetime.strftime(x, "%Y-%m-%d") for x in data11]
    result = pd.merge(data1, data2, on=['code', 'date'])
    print(len(result),result)
    result.to_csv(dir_yuce2, mode='w', index=False, header=True)

def factor_chuli_all(dir_data,dir_yuce4,dir_test):
    """
     对预测明细与因子进行合并，并进行筛选
     :param dir_data: 各个文件保存的目录
     :param ,dir_yuce4: 最终的需要的选股数据，筛选后的
     :return: 无
     """
    # 初始的全市场预测明细数据
    dir_yuce1 = os.path.join(dir_test, 'TestDataSet_predict.csv')
    # 合并后的全市场预测明细数据，与不需要二次处理的因子合并
    dir_yuce2 = os.path.join(dir_data, 'TestDataSet_predict2.csv')
    # 合并后的全市场预测明细数据，与需要二次处理的因子合并
    dir_yuce3 = os.path.join(dir_data, 'TestDataSet_predict3.csv')
    # 保存需要二次处理的因子数据
    save_yinzi = os.path.join(dir_data, 'shujuji2.csv')
    # 保存全市场股价涨停，因子数据
    save_yinzi2 = os.path.join(dir_data, 'shujuji3.csv')
    factor_chuli1(dir_yuce1, dir_yuce2, save_yinzi2,save_yinzi)
    factor_chuli2(dir_yuce2, save_yinzi, dir_yuce3)
    data_pre = pd.read_table(dir_yuce3, sep=',')  # 预测明细
    factor_chuli3(data_pre, dir_yuce4)

def read_pre2(dir_data,dir_test):
    """
   读取筛选与合并后的数据
   :param dir_data: 各个文件保存的目录
   :return: data_pre: 筛选与合并后的数据
   """
    #最终的筛选与合并后的预测明细数据
    dir_yuce4 = os.path.join(dir_data, 'TestDataSet_predict4.csv')
    factor_chuli_all(dir_data, dir_yuce4,dir_test)
    data_pre = pd.read_table(dir_yuce4, sep=',')  # 预测明细
    data_pre['code'] = [i[2:] for i in data_pre['code'].tolist()]
    #print(data_pre)
    return data_pre

# model_train() # 模型预测
mc.load_config_huice_1000()
dir_test = cn.trainFileName25 #预测明细读取的目录
print(dir_test)
#回测文件保存的目录
dir_data = r'E:\stock_simulation\Stock_date\model_trend'
data_xgboost = read_pre2(dir_data,dir_test)              # 将趋势因子数据插入到预测明细数据、并做趋势因子筛选，返回筛选后的结果集
