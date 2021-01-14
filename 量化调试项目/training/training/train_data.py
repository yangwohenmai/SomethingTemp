# -*- coding: utf-8 -*-
"""
 选择全市场市值前1500的股票
 读取qa导出的日线和分钟数据，计算因子，生成因子数据集用于训练

"""
import talib
from chart import extract_feature
import datetime as dt
import time
import numpy as np
import numpy
import os
import pandas as pd
import config2 as cn
from pandas import DataFrame  # DataFrame通常来装二维的表格
import re
import time as tm
import warnings
import multiprocessing
import tushare as ts
from tools import get_stock_all, get_top_longtou,get_stock_list,get_stock_all2
from qa_export import export_csv_files,get_trade_dates
from merge_30min import merge


def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) * childern_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def update_shindex(huice_shIndex_data_dir):
    # data = ts.get_k_data('000001', index=True, start='2008-01-01', end='')
    cons = ts.get_apis()

    data = ts.bar('000001', conn=cons, asset='INDEX', start_date='2008-01-01', end_date='').reset_index().sort_index(
        ascending=False)
    print(data)
    # data=ts.get_hist_data('sh').reset_index().sort_index(ascending=False)
    data.rename(columns={'datetime': 'date'})
    data.to_csv(huice_shIndex_data_dir, mode='w', index=False, header=True)
    print('end')


def get_date(stock_dict):
    """
    读取指数
    :param stock_dict:
    :return:
    """

    huice_shIndex_data_dir = stock_dict['huice_shIndex_data_dir']
    date_list = get_trade_dates('2008-01-01', str(dt.datetime.now().date()), save=True)[-1500:]
    data_sh = pd.read_table(huice_shIndex_data_dir, sep=',')  # 读取每日更新的上证指数

    #date_list = data_sh['date'].tolist()[-2000:]  # 得到日期列表
    data_sh['rise_down'] = data_sh['close'].pct_change() * 100
    data_sh = data_sh.round(2).dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    data_sh = data_sh[data_sh['date'].isin(date_list)]  # 得到上证指数涨跌幅


    return date_list, data_sh


def Min_factor_cal(min_data_30):
    qrr_list = []
    b2 = []
    qrr_num = 40  # .set_index('date')
    min_data_30 = min_data_30.reset_index()

    for indexs in range(len(min_data_30)):
        if indexs > qrr_num:  # 以一天的成交量之和来计算量比
            vol = min_data_30.loc[indexs]['vol']
            vol_mean = min_data_30[indexs - qrr_num:indexs]['vol'].mean()
            qrr = vol / vol_mean

            aa = min_data_30.loc[indexs]['vol'] + min_data_30.loc[indexs - 1]['vol']
            abb = min_data_30[indexs - 40:indexs]['vol'].mean()
            c = aa / abb
        else:
            qrr = 0
            c = 0
        b2.append(c)
        qrr_list.append(qrr)

    close = numpy.asarray(min_data_30['close'].tolist())
    open = numpy.asarray(min_data_30['open'].tolist())
    high = numpy.asarray(min_data_30['high'].tolist())
    low = numpy.asarray(min_data_30['low'].tolist())
    close_rets = (close[1:] / close[:-1] - 1) * 100
    close_rets = [0] + close_rets.tolist()

    min_data_30['量比'] = qrr_list
    min_data_30['30分钟涨跌幅'] = close_rets

    min_close_open = (close / open - 1) * 100
    min_low_open = (low / open - 1) * 100
    min_low_close = (close / low - 1) * 100
    min_high_open = (high / open - 1) * 100
    min_high_close = (close / high - 1) * 100

    date_open_close = (open[1:] / close[:-1] - 1) * 100
    date_open_close = [0] + date_open_close.tolist()

    rets = (close[2:] / close[:-2] - 1) * 100
    rets = rets.tolist()
    rets.insert(0, 0)
    rets.insert(0, 0)

    min_data_30['量比2'] = b2
    min_data_30['30分钟涨跌幅2'] = rets

    ma5 = numpy.nan_to_num(talib.MA(close, timeperiod=5))

    min_data_30['min_close_open'] = min_close_open
    min_data_30['min_low_open'] = min_low_open
    min_data_30['min_low_close'] = min_low_close
    min_data_30['min_high_open'] = min_high_open
    min_data_30['min_high_close'] = min_high_close
    min_data_30['date_open_close'] = date_open_close
    min_data_30['ma5_30'] = ma5

    return min_data_30


def cal_label(min_data_30,label):
    for each_label in range(18):
        min_data_30[label[each_label]] = ''
    for index in range(int(len(min_data_30) / 8) - 1):
        group_trade = min_data_30[index * 8:index * 8 + 8 * 2].reset_index(drop=True)  # 切割dataframe
        #print(11,group_trade)
        time_trade = group_trade.ix[[0, 1, 2, 6,8,9,10,14], [0,1,2,3,4,5]].reset_index(drop=True)
        #print(time_trade)
        if index == 0:
            limit_up0 = time_trade[0:1]['open'].tolist()[0]
        else:
            limit_up0 = min_data_30.ix[index * 8-1]['close']
        #print(limit_up0)
        label_1 = time_trade[0:1]['open'].tolist()[0]  # 9:30
        label_2 = time_trade[0:1]['close'].tolist()[0]   # 10:00
        label_3 = time_trade[1:2]['close'].tolist()[0]   # 10:30
        label_4 = time_trade[2:3]['close'].tolist()[0]   # 11:00
        label_5 = time_trade[3:4]['close'].tolist()[0]

        label_6 = time_trade[4:5]['open'].tolist()[0]
        label_7 = time_trade[4:5]['close'].tolist()[0]
        label_8 = time_trade[5:6]['close'].tolist()[0]
        label_9 = time_trade[6:7]['close'].tolist()[0]
        label_10 = time_trade[7:8]['close'].tolist()[0]

        label1 = (label_6 / label_1 - 1) * 100
        label2 = (label_7 / label_2 - 1) * 100
        label3 = (label_8 / label_3 - 1) * 100
        label4 = (label_9 / label_4 - 1) * 100
        label5 = (label_10 / label_5 - 1) * 100

        limit_up1 = (label_1 / limit_up0 - 1) * 100
        limit_up2 = (label_2 / limit_up0 - 1) * 100
        limit_up3 = (label_3 / limit_up0 - 1) * 100
        limit_up4 = (label_4 / limit_up0 - 1) * 100
        limit_up5 = (label_5 / limit_up0 - 1) * 100

        label3_5 = []
        for days in [3,5]:
            group_trade2 = min_data_30[index * 8:index * 8 + 8 * (1 + days)].reset_index(drop=True)  # 切割dataframe  1
            #print(group_trade2)
            kl2 = group_trade2[1:2]['close'].tolist()[0]
            we = group_trade2[-7:-6] ['close'].tolist()[0]
            we2 = np.max(group_trade2[2:]['high'].tolist())
            lab = (we - kl2) / kl2 * 100
            lab2 = (we2 - kl2) / kl2 * 100
            label3_5 = label3_5+[lab,lab2]


        label_all = [label1, label2, label3, label4,label5]+ label3_5+\
                    [limit_up1,limit_up2,limit_up3,limit_up4,limit_up5] + [limit_up3]*4
        #print(label_all) '涨停1', '涨停2', '涨停3', '涨停4','涨停5'
        for n in range(18):
            min_data_30.loc[(index) * 8 + 1, [label[n]]] = label_all[n]
    return min_data_30



def cal_feature30(min_data_30,filename,label):
    t1 = tm.time()
    # 计算各个标签值 ,标签频率为9:30,10:00,10:30,11:00,14:30

    # print(min_data_30)
    min_data_30 = cal_label(min_data_30, label)
    # print(min_data_30)

    t2 = tm.time()
    # min_data_30.round(4).to_csv('AA0.csv', mode='w', index=False, header=True)
    # print("1:{0}".format(t2 - t1))

    # 计算涨跌幅与量比
    min_data_30 = Min_factor_cal(min_data_30)
    #min_data_30.round(4).to_csv('AA1.csv', mode='w', index=False, header=True)
    t3 = tm.time()
    # print("2:{0}".format(t3 - t2))

    # 各个因子的取名，总共8个点
    # 对早盘与晚盘和标签值的数据转化为指定的格式
    name = ['前一日尾盘30分钟涨幅', '前一日尾盘30分钟量比',
            '当日早盘30分钟涨幅', '当日早盘30分钟量比', '前一日尾盘60分钟涨幅', '前一日尾盘60分钟量比'
        , '两日60分钟涨幅', '两日60分钟量比', '当日早盘60分钟涨幅',
            'min_close_open', 'min_low_open', 'min_low_close', 'min_high_open', 'min_high_close',
            '当日早盘60分钟量比', '当日早盘后30分钟涨幅', '当日早盘后30分钟量比',
            'min_close_open2', 'min_low_open2', 'min_low_close2', 'min_high_open2', 'min_high_close2',
            'date_open_close',
            'close30', 'high00', 'high30', 'close15', 'open00','ma5_10_00','ma5_10_30','ma2_5_10_00','ma2_5_10_30','price_10_00','price_10_30']

    min_num_15 = ['min' + str(i) for i in range(8)]
    time_min_15_1 = [i + '30分钟涨跌幅' for i in min_num_15]
    time_min_15_2 = [i + '30分钟量比' for i in min_num_15]
    time_min_15_3 = [i + 'min_close_open' for i in min_num_15]
    time_min_15_4 = [i + 'min_low_open' for i in min_num_15]
    time_min_15_5 = [i + 'min_high_open' for i in min_num_15]
    time_min_15_6 = [i + 'min_high_close' for i in min_num_15]

    time_min_15_7 = [i + '30分钟涨跌幅2' for i in min_num_15]
    time_min_15_8 = [i + '30分钟量比2' for i in min_num_15]

    name2 = ['代码', '日期'] + name + time_min_15_1 + time_min_15_2 + time_min_15_3 + \
            time_min_15_4 + time_min_15_5 + time_min_15_6 + time_min_15_7 + time_min_15_8 + label
    result = DataFrame(columns=(name2))
    grade_num_list1 = [num for num in range(0, len(min_data_30), 8)]
    for grade_num in range(8):
        grade_num_list = [num + grade_num for num in grade_num_list1]
        # print(grade_num)
        data = min_data_30.loc[grade_num_list, :]
        result['min' + str(grade_num) + '30分钟涨跌幅'] = data['30分钟涨跌幅'].tolist()[:-1]
        result['min' + str(grade_num) + '30分钟量比'] = data['量比'].tolist()[:-1]
        result['min' + str(grade_num) + 'min_close_open'] = data['min_close_open'].tolist()[:-1]
        result['min' + str(grade_num) + 'min_low_open'] = data['min_low_open'].tolist()[:-1]
        result['min' + str(grade_num) + 'min_high_open'] = data['min_high_open'].tolist()[:-1]
        result['min' + str(grade_num) + 'min_high_close'] = data['min_high_close'].tolist()[:-1]

        result['min' + str(grade_num) + '30分钟涨跌幅2'] = data['30分钟涨跌幅'].tolist()[1:]
        result['min' + str(grade_num) + '30分钟量比2'] = data['量比'].tolist()[1:]

        if grade_num == 1:  # 代码，日期，标签都在这一行
            # print(data)
            result['日期'] = data['日期'].tolist()[1:]
            result['代码'] = [filename[:8]] * len(result['日期'].tolist())
            result['涨跌幅1'] = data['涨跌幅1'].tolist()[1:]
            result['涨跌幅2'] = data['涨跌幅2'].tolist()[1:]
            result['涨跌幅3'] = data['涨跌幅3'].tolist()[1:]
            result['涨跌幅4'] = data['涨跌幅4'].tolist()[1:]
            result['涨跌幅5'] = data['涨跌幅5'].tolist()[1:]
            result['涨跌幅6'] = data['涨跌幅6'].tolist()[1:]
            result['涨跌幅7'] = data['涨跌幅7'].tolist()[1:]
            result['涨跌幅8'] = data['涨跌幅8'].tolist()[1:]
            result['涨跌幅9'] = data['涨跌幅9'].tolist()[1:]
            result['涨停1'] = data['涨停1'].tolist()[1:]
            result['涨停2'] = data['涨停2'].tolist()[1:]
            result['涨停3'] = data['涨停3'].tolist()[1:]
            result['涨停4'] = data['涨停4'].tolist()[1:]
            result['涨停5'] = data['涨停5'].tolist()[1:]
            result['涨停6'] = data['涨停6'].tolist()[1:]
            result['涨停7'] = data['涨停7'].tolist()[1:]
            result['涨停8'] = data['涨停8'].tolist()[1:]
            result['涨停9'] = data['涨停9'].tolist()[1:]

    data1 = min_data_30[min_data_30['时间2'] == '10:30:00']
    data2 = min_data_30[min_data_30['时间2'] == '15:00:00']
    data3 = min_data_30[min_data_30['时间2'] == '10:00:00']


    result['前一日尾盘60分钟涨幅'] = data2['30分钟涨跌幅2'].tolist()[:-1]
    result['前一日尾盘60分钟量比'] = data2['量比2'].tolist()[:-1]

    result['前一日尾盘30分钟涨幅'] = data2['30分钟涨跌幅'].tolist()[:-1]
    result['前一日尾盘30分钟量比'] = data2['量比'].tolist()[:-1]

    result['两日60分钟涨幅'] = data3['30分钟涨跌幅2'].tolist()[1:]
    result['两日60分钟量比'] = data3['量比2'].tolist()[1:]

    result['close30'] = data1['close'].tolist()[1:]  # 10点半的收盘价
    result['high00'] = data3['high'].tolist()[1:]  # 10点的最高价
    result['high30'] = data1['high'].tolist()[1:]  # 10点半的最高价
    result['close15'] = data2['close'].tolist()[1:]  # 10点的最高价
    result['open00'] = data3['open'].tolist()[1:]  # 10点半的最高价

    a = 'True'
    if a == 'True':
        result['当日早盘30分钟涨幅'] = data3['30分钟涨跌幅'].tolist()[1:]
        result['当日早盘30分钟量比'] = data3['量比'].tolist()[1:]
        result['当日早盘60分钟涨幅'] = data1['30分钟涨跌幅2'].tolist()[1:]
        result['当日早盘60分钟量比'] = data1['量比2'].tolist()[1:]
        result['当日早盘后30分钟涨幅'] = data1['30分钟涨跌幅'].tolist()[1:]
        result['当日早盘后30分钟量比'] = data1['量比'].tolist()[1:]
        result['min_close_open'] = data3['min_close_open'].tolist()[1:]
        result['min_low_open'] = data3['min_low_open'].tolist()[1:]
        result['min_low_close'] = data3['min_low_close'].tolist()[1:]
        result['min_high_open'] = data3['min_high_open'].tolist()[1:]
        result['min_high_close'] = data3['min_high_close'].tolist()[1:]

        result['min_close_open2'] = data1['min_close_open'].tolist()[1:]
        result['min_low_open2'] = data1['min_low_open'].tolist()[1:]
        result['min_low_close2'] = data1['min_low_close'].tolist()[1:]
        result['min_high_open2'] = data1['min_high_open'].tolist()[1:]
        result['min_high_close2'] = data1['min_high_close'].tolist()[1:]
        result['date_open_close'] = data3['date_open_close'].tolist()[:-1]

# 'ma5_10_00','ma5_10_30','price_10_00','price_10_30' 'ma2_5_10_00','ma2_5_10_30'

        result['ma5_10_00'] = data3['ma5_30'].tolist()[1:]
        result['ma5_10_30'] = data1['ma5_30'].tolist()[1:]

        result['ma2_5_10_00'] = data3['ma5_30'].tolist()[:-1]
        result['ma2_5_10_30'] = data1['ma5_30'].tolist()[:-1]

        result['price_10_00'] = data3['close'].tolist()[1:]
        result['price_10_30'] = data1['close'].tolist()[1:]

    t4 = tm.time()
    # result.round(4).to_csv('AA2.csv', mode='w', index=False, header=True)
    # print (result)
    # print("3:{0}".format(t4 - t1))
    return result



def mor_late(data):
    b = []
    b2 = []
    for indexs in range(len(data)):
        if indexs > 40:
            aa = data.loc[indexs]['vol'] + data.loc[indexs - 1]['vol']
            abb = data[indexs - 40:indexs]['vol'].mean()
            c = aa / abb

            aa2 = data.loc[indexs]['vol']
            abb2 = data[indexs - 40:indexs]['vol'].mean()
            c2 = aa2 / abb2
        else:
            c = 0
            c2 = 0
        b.append(c)
        b2.append(c2)

    hj = numpy.asarray(data['close'].tolist())
    hj_open = numpy.asarray(data['open'].tolist())
    hj_high = numpy.asarray(data['high'].tolist())
    hj_low = numpy.asarray(data['low'].tolist())
    rets = (hj[2:] / hj[:-2] - 1) * 100
    rets = rets.tolist()
    rets.insert(0, 0)
    rets.insert(0, 0)

    data['量比'] = b
    data['个股涨跌幅2'] = rets

    rets2 = (hj[1:] / hj[:-1] - 1) * 100
    rets2 = rets2.tolist()
    rets2.insert(0, 0)
    min_close_open = (hj / hj_open - 1) * 100
    min_low_open = (hj_low / hj_open - 1) * 100
    min_low_close = (hj / hj_low - 1) * 100
    min_high_open = (hj_high / hj_open - 1) * 100
    min_high_close = (hj / hj_high - 1) * 100

    date_open_close = (hj_open[1:] / hj[:-1] - 1) * 100
    date_open_close = date_open_close.tolist()
    date_open_close.insert(0, 0)

    data['量比2'] = b2
    data['个股涨跌幅22'] = rets2  # 'min_close_open', 'min_low_open', 'min_low_close', 'min_high_open', 'min_high_close'

    data['min_close_open'] = min_close_open
    data['min_low_open'] = min_low_open
    data['min_low_close'] = min_low_close
    data['min_high_open'] = min_high_open
    data['min_high_close'] = min_high_close
    data['date_open_close'] = date_open_close

    return data

    # abc = [str + '.csv' for str in abc]
    # return abc


def init_parameter():
    stock_dict = {}
    stock_dict['huice_data_dir'] = cn.huice_data_dir  # 因子数据集目录
    stock_dict['huice_data_dir2'] = cn.huice_data_dir2  # 因子数据集子目录
    stock_dict['huice_shIndex_data_dir'] = cn.huice_shIndex_data_dir
    stock_dict['mor_Stock_price'] = cn.mor_Stock_price
    stock_dict['huice_rise_down_dir'] = cn.huice_rise_down_dir
    stock_dict['huice_shujuji_dir'] = cn.huice_shujuji_dir
    stock_dict['his_min_data_dir'] = cn.qa_his_min_data_dir  # 原始数据集30min
    stock_dict['his_day_data_dir'] = cn.qa_his_day_data_dir  # 原始数据集日线
    stock_dict['selector'] = cn.selector  # 因子集
    stock_dict['window'] = cn.window  # 时间窗口

    # 清理相关目录
    huice_data_dir2 = stock_dict['huice_data_dir2']
    huice_rise_down_dir = stock_dict['huice_rise_down_dir']
    huice_shujuji_dir = stock_dict['huice_shujuji_dir']
    mor_Stock_price = stock_dict['mor_Stock_price']

    if os.path.exists(huice_rise_down_dir):
        os.remove(huice_rise_down_dir)
    if os.path.exists(huice_shujuji_dir):
        os.remove(huice_shujuji_dir)
    if os.path.exists(mor_Stock_price):
        os.remove(mor_Stock_price)
    for i in os.listdir(huice_data_dir2):
        name = os.path.join(huice_data_dir2, i)
        if os.path.exists(name):
            os.remove(name)

    return stock_dict


def mycallback(df_shuju):
    try:
        df3 = df_shuju['df']
        huice_data_dir = df_shuju['huice_shujuji_dir']
        if os.path.exists(huice_data_dir):
            header = False
        else:
            header = True
        df3.round(2).to_csv(huice_data_dir, mode='a', index=False, header=header)
    except:
        pass


# 初始化删除数据集
def del_shuju(stock_dict):
    huice_data_dir2 = stock_dict['huice_data_dir2']
    huice_rise_down_dir = stock_dict['huice_rise_down_dir']
    huice_shujuji_dir = stock_dict['huice_shujuji_dir']
    mor_Stock_price = stock_dict['mor_Stock_price']

    if os.path.exists(huice_rise_down_dir):
        os.remove(huice_rise_down_dir)
    if os.path.exists(huice_shujuji_dir):
        os.remove(huice_shujuji_dir)
    if os.path.exists(mor_Stock_price):
        os.remove(mor_Stock_price)
    for i in os.listdir(huice_data_dir2):
        name = os.path.join(huice_data_dir2, i)
        if os.path.exists(name):
            os.remove(name)


# 把子进程生产的数据，合并为一份数据集
def integrate_data(stock_dict):
    huice_data_dir2 = stock_dict['huice_data_dir2']
    huice_shujuji_dir = stock_dict['huice_shujuji_dir']
    print(huice_data_dir2)
    print(huice_shujuji_dir)
    for i in os.listdir(huice_data_dir2):
        name = os.path.join(huice_data_dir2, i)
        all_data = pd.read_table(name, sep=',')
        if os.path.exists(huice_shujuji_dir):
            header = False
        else:
            header = True
        all_data.to_csv(huice_shujuji_dir, mode='a', index=False, header=header)

    # 最后删除子进程生成的数据
    for i in os.listdir(huice_data_dir2):
        name2 = os.path.join(huice_data_dir2, i)
        if os.path.exists(name2):
            os.remove(name2)


def build_train_data():
    """
    读取qa导出的日线和分钟数据，计算因子，构造因子数据集用于训练
    :return:
    """
    # 选择全市场的股票
    all_code = get_stock_all()  # get_top_longtou() #[:10] # get_top_longtou() #get_stock_all() #get_stock_list(sort='mktcap', len=4000, update=False)
    all_code = all_code[:20]
    print('股票数量:', len(all_code))
    #print(all_code)

    if 'sh603429.csv' in all_code:
        print(123456)
    else:
        print(6666)

    # 参数初始化，清理相关目录
    stock_dict = init_parameter()
    # integrate_data(stock_dict)  # 把子进程生产的数据，合并为一份数据集
    # exit()
    date_list, shIndex_chglist = get_date(stock_dict)  # 得到日期列表，可以变为开始日期与结束日期
    print('日期间隔', len(date_list), date_list[0], date_list[-1])


    # 开多进程计算多因子
    qa_min_data_dir = stock_dict['his_min_data_dir']
    qa_day_data_dir = stock_dict['his_day_data_dir']
    SELECTOR = stock_dict['selector']
    window = stock_dict['window']
    mainStart = time.time()  # 记录主进程开始时间

    p = multiprocessing.Pool(2)  # 开辟进程池
    for i in range(len(all_code)):  # 开辟14个进程
        stock_code = all_code[i]
        # load_data(stock_code, qa_min_data_dir, qa_day_data_dir, SELECTOR, window, date_list, shIndex_chglist,
        #           stock_dict, i)
        p.apply_async(load_data, (stock_code, qa_min_data_dir,qa_day_data_dir,SELECTOR,window,
                                        date_list, shIndex_chglist, stock_dict, i,), callback=mycallback)

    p.close()  # 关闭进程池
    p.join()  # 等待开辟的所有进程执行完后，主进程才继续往下执行
    print('All subprocesses done')
    mainEnd = time.time()  # 记录主进程结束时间
    print('All process ran %0.2f seconds.' % (mainEnd - mainStart))  # 主进程执行时间

    integrate_data(stock_dict)  # 把子进程生产的数据，合并为一份数据集
    #mor_Stock_price(stock_dict['huice_data_dir'])  # 早盘涨停数据更新


def mor_Stock_price(dataset_dir92):
    dataset_dir93 = os.path.join(dataset_dir92, 'shujuji.csv')
    dataset_dir24 = os.path.join(dataset_dir92, 'mor_Stock_price.csv')

    data = pd.read_csv(dataset_dir93, sep=',')
    df16 = data.ix[::, ['daima', 'data', '涨停1', '涨停2', '涨停3', '涨停4', '涨停5',
                        '涨停6', '涨停7', '涨停8', '涨停9']]
    df16.to_csv(dataset_dir24, mode='w', index=False, header=True)


def up_down2():
    dataset_dir24 = r'E:\new file\lstm\shujuji\zhangdiefu21.csv'
    dataset_dir25 = r'E:\new file\lstm\shujuji\zhangdiefu3.csv'
    if os.path.exists(dataset_dir25):
        data2 = pd.read_table(dataset_dir25, sep=',', index_col=0).T
        if os.path.exists(dataset_dir24):
            data = pd.read_table(dataset_dir24, sep=',', index_col=0)
        else:
            data = DataFrame(columns=(data2.columns))

        a = set(data.index)
        b = set(data2.index)
        c = list(b - a)
        df = data2.ix[c, ::].sort_index(ascending=True)
        # print (df)
        df.to_csv(dataset_dir24, mode='a', encoding='utf-8', index=True, header=True)
        os.remove(dataset_dir25)


def up_down(data, xulie, filename, stock_dict):
    huice_rise_down_dir = stock_dict['huice_rise_down_dir']
    xulie2 = data['data'].tolist()
    up_down = data['涨跌幅1'].tolist()
    prestock_dict2 = DataFrame(index=[filename[0:8]], columns=(xulie2))
    prestock_dict2.loc[filename[0:8]] = up_down
    df = DataFrame(columns=(xulie))
    df = pd.concat([df, prestock_dict2], sort=False)
    df = df.ix[::, :-1]
    if os.path.exists(huice_rise_down_dir):
        header = False
    else:
        header = True
    df.to_csv(huice_rise_down_dir, mode='a', encoding='utf-8', index=True, header=header)


# 数据清洗，确保数据格式正确
def date_clean(min_data, xulie):
    y = min_data['datetime'].str.split(r' ', expand=True)
    y.columns = ['日期', '时间2']
    min_data = pd.concat([min_data, y], axis=1)
    min_data = min_data[min_data['日期'].isin(xulie)].reset_index()  # min

    group_data = min_data.groupby('日期')
    group_data2 = group_data.groups
    # print (group_data.values)
    b = []
    for k in group_data2:  # 这种方式效率比较高
        a = len(group_data2[k])
        if a == 8:
            b.append(k)
    min_data = min_data[min_data['日期'].isin(b)].reset_index(drop=True)
    return min_data


def load_data(stock_code, qa_min_data_dir,qa_day_data_dir,SELECTOR,window, date_list, shIndex_chglist, stock_dict, i):
    """
      读取股票的历史日线和30min数据,进行因子计算得到数据集
      :param stock_code:
      :param qa_day_data_dir:    历史日线数据目录
      :param qa_min_data_dir:    历史分钟数据目录
      :param date_list:       数据集的日期范围列表
      :param shIndex_chglist: 日期范围内的上涨涨幅列表
      :param selector:        因子集合
      :param data_dir2:       子进程数据集的保存目录
      :return:
      """
    # try:
    # if filename != 'sz000511.csv':
    print(stock_code, i)

    his_min_data = os.path.join(qa_min_data_dir, stock_code[0:8].upper() + '.csv')
    his_day_data = os.path.join(qa_day_data_dir, stock_code[0:8].upper() + '.csv')  #

    min_data = pd.read_table(his_min_data, sep=',')  # 标签与盘中因子
    day_data = pd.read_table(his_day_data, sep=',')  # 基础数据

    min_data = date_clean(min_data, date_list)  # 数据清洗，确保数据格式正确
    label = ['涨跌幅1', '涨跌幅2', '涨跌幅3', '涨跌幅4', '涨跌幅5',
             '涨跌幅6', '涨跌幅7', '涨跌幅8', '涨跌幅9', '涨停1', '涨停2', '涨停3', '涨停4', '涨停5',
             '涨停6', '涨停7', '涨停8', '涨停9']
    min_data = cal_feature30(min_data, stock_code,label)  # 计算盘中因子

    day_data = day_data[day_data['datetime'].isin(date_list)]  # data
    day_list = set(day_data['datetime'].tolist())
    min_list_30 = set(min_data['日期'].tolist())
    date_list = day_list & min_list_30

    if len(date_list) > 0:
        min_data = min_data[min_data['日期'].isin(date_list)]  # min
        day_data = day_data[day_data['datetime'].isin(date_list)]  # data
        shIndex_chglist = shIndex_chglist[shIndex_chglist['date'].isin(date_list)]  # 上证指数
        stock_code = stock_code[0:2].upper() + stock_code[2:8]

        moving_features, moving_labels, x2 = extract_feature(raw_data=min_data, raw_data2=day_data, raw_data3=shIndex_chglist,
                                                             selector=SELECTOR, window=window, with_label=True,
                                                             flatten=True, xulie=date_list, filename=stock_code)
        # print (moving_features.shape)
        result = DataFrame(moving_features)
        result2 = DataFrame(moving_labels)
        result2.columns = label

        df3 = pd.concat([result, result2], axis=1)
        df3 = pd.DataFrame(df3)
        df3.insert(0, 'data', x2[1, ::].tolist())
        df3.insert(0, 'daima', x2[0, ::].tolist())
        df3 = df3.iloc[:-1, ::]
        # print(df3)

        #up_down(df3, xulie, filename, stock_dict)

        df_shuju = {}
        df_shuju['df'] = df3
        huice_data_dir = stock_dict['huice_data_dir2']
        subfile_file = os.path.join(huice_data_dir, multiprocessing.current_process().name + '.csv')
        df_shuju['huice_shujuji_dir'] = subfile_file

        return df_shuju
    # except:
    # print('error',filename,i)


if __name__ == '__main__':
    t1 = tm.time()
    warnings.filterwarnings("ignore")

    # 从QA本地数据库导出日线和30min数据文件
    min_data_dir = cn.qa_his_min_data_dir  # 盘后分钟数据存放目录
    day_data_dir = cn.qa_his_day_data_dir  # 日线数据存放目录
    print(min_data_dir)
    export_csv_files('2020-01-01', '2020-12-28', min_data_dir, day_data_dir, clean=True)
    # # 对接淘宝30min数据
    # merge()
    # 读取qa导出的日线和分钟数据，计算因子，构造因子数据集用于训练
    build_train_data()

    t2 = tm.time()
    print("总耗时:{0}".format(t2 - t1))

    exit(0)