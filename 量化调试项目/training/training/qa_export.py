# encoding:utf-8
# 从QA本地数据库导出全市场指定时间段的日线和30min数据到csv文件

import tushare as ts
import numpy as np
import pandas as pd
import os
import warnings
import time as tm
import datetime
import config2 as cn
import QUANTAXIS as QA
import tushare.stock.fundamental as fd
from tools import get_stock_all
import merge_30min as merge_30min


# 删除指定目录中所有子目录和文件
def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


# 排除ST的股票
def ST(abc):
    df = fd.get_stock_basics()
    df = df.ix[df.name.str.contains('S')]

    d = set(abc) - (set(abc) & set(df.index))

    n = []
    for i in d:
        if i[:2] == '60':
            m = 'sh' + i
        elif i[:2] == '00' or i[:2] == '30':
            m = 'sz' + i
        n.append(m)
    return n




# 从QA本地数据库导出日线和30min数据文件
def export_csv_files(start, end, min_data_dir, day_data_dir, clean=False):
    if clean:
        del_file(day_data_dir)
        del_file(min_data_dir)

    #上证指数的更新，返回start~end区间的交易日列表
    date_list = get_trade_dates(start, end)

    tstart = tm.time()

    # 得到全市场的股票代码
    stock_file_list = get_stock_all() #get_stock_list(sort='mktcap', len=4000)  #get_all_stockcode0()

    print('开始从QA导出股票数据...', len(stock_file_list))

    round = 1       #下载的轮次
    while len(stock_file_list) > 0:
        failed_code_list = []
        for stock_file in stock_file_list:
            print("  export stock file: " + stock_file)

            flag1 = qa_export_day(stock_file, date_list, day_data_dir)     # 日线数据保存到csv
            flag2 = qa_export_30min(stock_file, date_list, min_data_dir)    # 30分钟数据保存到csv
            #flag2 = True
            if flag1 is False or flag2 is False:
                print("{0}: {1}".format(stock_file, 'error'))
                failed_code_list.append(stock_file)

        if round >= 2 or len(failed_code_list) == 0:
            break
        else:
            stock_file_list = failed_code_list
            tm.sleep(100)
            print('第 {0} 轮下载完成'.format(round))
            round += 1

    tend = tm.time()
    print("QA导出完成   总耗时：{0}".format(tend - tstart))


def get_trade_dates(start, end, save=False):
    """
    更新上证指数并返回start~end区间内的交易日期列表
    :param start:
    :param end:
    :param save: 是否写入到指数文件
    :return:
    """
    data = QA.QA_fetch_index_day_adv('000001', start, end).data.reset_index()
    if save:
        data.to_csv(cn.huice_shIndex_data_dir, mode='w', index=False, header=True)
    data['date'] = data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    return data['date'].tolist()


def qa_export_day(stock_file, date_list, day_data_dir):
    """
    从QA本地数据库中获取日线数据，并保存到csv
    :param stock_file:   eg: sh600001.csv
    :param date_list:
    :return:
    """
    try:
        day_data = QA.QA_fetch_stock_day_adv(stock_file[2:8], date_list[0], date_list[-1]).to_qfq().data.reset_index()
        block = QA.QAAnalysis_block(code=[stock_file[2:8]], start=date_list[0], end=date_list[-1])
        tur = block.stock_turnover().reset_index()
        day_data = pd.merge(day_data, tur, on=['date', 'code'])
        # print(day_data)
        # print(day_data.columns)
        day_data.rename(columns={"date": "datetime", "volume": "vol", 0: "tor"}, inplace=True)
        # day_data.columns = ['datetime', 'code', 'open', 'high', 'low', 'close', 'vol', 'amount', 'preclose','adj','tor']

        day_data = day_data[day_data.vol > 0.5]  # 排除停牌的股票，成交量~=0

        if day_data is None:
            return False

        # 日线数据保存
        day_data_file = os.path.join(day_data_dir, stock_file)
        if len(day_data) > 0:
            day_data.to_csv(day_data_file, mode='w', index=False, header=True)  # 加载盘后数据

    except:
        return False
    return True


def qa_export_30min(stock_file, date_list, min_data_dir):
    """
    从QA本地数据库中获取30min数据，并保存到csv
    :param stock_file: 6位数股票代码
    :param date_list:
    :return:
    """
    try:

        min_data = QA.QA_fetch_stock_min_adv(stock_file[2:8], date_list[0], date_list[-1], '30min').to_hfq().data.reset_index()
        min_data.columns = ['datetime', 'code', 'open', 'high', 'low', 'close', 'vol', 'amount', 'preclose',
                            'type']
        min_data['vol'] = min_data['vol']/100

        if min_data is None:
            return False

        # 如果数据没问题，则保存
        min_data = min_data.ix[:, ['datetime', 'code', 'open', 'close', 'high', 'low', 'vol']]
        min_data_file = os.path.join(min_data_dir, stock_file)
        if len(min_data) > 0 :

            min_data.to_csv(min_data_file, mode='w', index=False, header=True)

    except:
        # print 'traceback.print_exc():'; traceback.print_exc()
        return False
    return True

#30min数据清洗，确保数据格式正确
def date_clean(data_all):
    y=data_all['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
    data = pd.concat([data_all, y], axis=1)
    data.columns = ['datetime', 'code', 'open', 'close', 'high', 'low', 'vol', '日期']
    group_data = data.groupby('日期')
    group_data2 = group_data.groups
    b = []
    num = 0
    state = True # 当数据正常是为True ，否则为False
    for k in group_data2:  # 这种方式效率比较高
        num += 1
        a = len(group_data2[k])
        if a == 8:
            b.append(k)
        else:                       # 如果数据出现缺失值，则退出
            print(k)
            state = False
    return state

if __name__ == '__main__':
    min_data_dir = cn.qa_his_min_data_dir                   # 盘后分钟数据存放目录
    day_data_dir = cn.qa_his_day_data_dir                   # 日线数据存放目录
    print(min_data_dir)
    # 从QA本地数据库导出日线和30min数据文件
    export_csv_files('2020-01-01', '2020-12-28', min_data_dir, day_data_dir, clean=True)

    # 对接淘宝30min数据
    # merge_30min.merge()
