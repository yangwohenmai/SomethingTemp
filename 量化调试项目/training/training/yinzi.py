import time as tm
import time
import warnings
import numpy as np
import numpy
import os
from numpy import newaxis
import pandas as pd
import copy
def cal_factor_days(name,data_copy,series,num1,num2):
    factor = []
    for i in range(len(series)):
        if i >= num1:
            b = sum(series[i - num1+1:i + 1])
            if b >= num2:
                l1 = 1
            else:
                l1 = 0
        else:
            l1 = 0
        factor.append(l1)

    data_copy[name] = factor
    return data_copy

def Factor1(data_copy,variable,num1,num2):
    """
    MA5>MA10，10日内出现的次数>=7
    :param variable: (MA5 - MA10)> variable
    :param num1,num2: num1日内出现的次数 >= num2
    :return: data_copy
    """
    name = 'factor1'
    factor1 = data_copy['128'].tolist()
    ma5_101 = [1 if factor1[i] > variable else 0 for i in range(len(factor1))]
    data_copy = cal_factor_days(name,data_copy, ma5_101, num1, num2)
    return data_copy

def Factor2(data_copy,variable1,variable2,num1,num2):
    """
    0<(MA5 - 昨日MA5)*100/昨日MA5<0.6，10日内出现的次数>=7
    :param variable1,variable2: variable1<(MA5 - 昨日MA5)*100/昨日MA5<variable2
    :param num1,num2: num1日内出现的次数 >= num2
    :return: data_copy
    """
    name = 'factor2'
    factor2 = data_copy['129'].tolist()
    ma5_102 = [1 if factor2[i] > variable1 and factor2[i] < variable2 else 0 for i in range(len(factor2))]
    data_copy = cal_factor_days(name,data_copy, ma5_102, num1, num2)
    return data_copy

def Factor3(data_copy,variable1,variable2,num1,num2):
    """
    0<(MA10 - 昨日MA10)*100/昨日MA10<0.4，10日内出现的次数>=7
    :param variable1,variable2: variable1<(MA5 - 昨日MA5)*100/昨日MA5<variable2
    :param num1,num2: num1日内出现的次数 >= num2
    :return: data_copy
    """
    name = 'factor3'
    factor3 = data_copy['130'].tolist()
    ma5_103 = [1 if factor3[i] > variable1 and factor3[i] < variable2 else 0 for i in range(len(factor3))]
    data_copy = cal_factor_days(name,data_copy, ma5_103, num1, num2)
    return data_copy

def Factor4(data_copy,variable,num1,num2):
    """
    CLOSE>MA10，10日内出现的次数>=6
    :param variable1: CLOSE-MA10 > variable
    :param num1,num2: num1日内出现的次数 >= num2
    :return: data_copy
    """
    name = 'factor4'
    factor4 = data_copy['131'].tolist()
    ma10_4 = [1 if factor4[i] > variable else 0 for i in range(len(factor4))]
    data_copy = cal_factor_days(name,data_copy, ma10_4, num1, num2)
    return data_copy

def Factor5(data_copy,variable1,variable2,num1,num2):
    """
    -3%<收盘涨幅<3%，10日内出现的次数>=7
    :param variable1,variable2: variable1<收盘涨幅<variable2
    :param num1,num2: num1日内出现的次数 >= num2
    :return: data_copy
    """
    name = 'factor5'
    factor5 = data_copy['132'].tolist()
    ma5_105 = [1 if factor5[i] > variable1 and factor5[i] < variable2 else 0 for i in range(len(factor5))]
    data_copy = cal_factor_days(name,data_copy, ma5_105, num1, num2)
    return data_copy

def Factor6(data_copy,variable):
    """
    (近10日内最高价-近10日最低价)/近10日最低价<15%
    :param variable: (近10日内最高价-近10日最低价)/近10日最低价<variable
    :return: data_copy
    """
    factor6 = data_copy['133'].tolist()
    ma5_106 = [1 if  factor6[i] < variable else 0 for i in range(len(factor6))]
    data_copy['factor6'] = ma5_106
    return data_copy

def Factor7(data_copy,variable):
    """
    (CLOSE-近5日最低价)/近5日最低价<10%
    :param variable: (CLOSE-近5日最低价)/近5日最低价<variable
    :return: data_copy
    """
    factor7 = data_copy['134'].tolist()
    ma5_107 = [1 if  factor7[i] < variable else 0 for i in range(len(factor7))]
    data_copy['factor7'] = ma5_107
    return data_copy

def Factor8(data_copy,variable,num1,num2):
    """
    10日内出现一次量比>=2（可选条件）
    :param variable1,variable2: variable1<收盘涨幅<variable2
    :param num1,num2: num1日内出现的次数 >= num2
    :return: data_copy
    """
    name = 'factor8'
    factor5 = data_copy['135'].tolist()
    ma5_105 = [1 if  factor5[i] >= variable else 0 for i in range(len(factor5))]
    data_copy = cal_factor_days(name,data_copy, ma5_105, num1, num2)
    return data_copy



def yinzi(yinzi,data):
    if yinzi == '':
        df_all = data.iloc[::, 0:2]
    elif yinzi=='MACD':
        df1 = data[data['9'] > 0].reset_index(drop=True).round(4)
        df_all = df1.iloc[::, 0:2]
    elif yinzi[:4]=='ROCP':
        a = 4          # 2=>3,4
        df111 = set(data[data['40'] > 7].index.tolist())
        df1112 = set(data[data['40'] <20].index.tolist())
        df222 = set(data[data['48'] >12].index.tolist())

        df99 = set(data[data['31'] > 3].index.tolist())

        df991 = set([i + 1 for i in df99])
        df992 = set([i + 2 for i in df99])
        df666 = set(data[data['25'] > 1].index.tolist())

        df1 = set(data[data['0'] > 2].index.tolist())           # 1=>2
        df12 = set(data[data['1'] > a].index.tolist())
        df13 = set(data[data['31'] > a].index.tolist())
        df14 = set(data[data['3'] > 6].index.tolist())
        df15 = set(data[data['4'] > a].index.tolist())
        df3 = set([i + 1 for i in df1])
        df2 = set([i + 2 for i in df1])

        df32 = set([i + 1 for i in df12])
        df22 = set([i + 2 for i in df12])

        df33 = set([i + 1 for i in df13])
        df23 = set([i + 2 for i in df13])

        df34 = set([i + 1 for i in df14])
        df24 = set([i + 2 for i in df14])

        df35 = set([i + 1 for i in df15])
        df25 = set([i + 2 for i in df15])
        # df77=df1 | df12 | df13 | df14 | df3 | df2 | df32 | df22 | df33 | df23 | df34 | df24 | df15 | df35 | df25| df99|df991|df992
        df77 = df12 | df13  | df32  | df33  | df15 | df35 | df99 | df34| df14 | df25 |df3|df2|df991|df992 |df1
        # df77 = df13  | df33  #&df666  #| df222
        df7 = df77 # -df9 -df_rise -df_rise_down_30-df_down4-df_down3 -df_down5 #df77
        df7 = [i for i in df7 if i>0 and i<len(data)]

        #df7=df7&df98
        #print (len(df77),len(df7),len(df9),len(data))
        df7 = list(df7)
        df_all = data.ix[df7, [0, 1]]
    elif yinzi == 'OP_CL':
        df9 = set(data[data['30'] <= 4.9].index.tolist())
        df91 = set(data[data['30'] >= -9.8].index.tolist())
        df92 = set(data[data['2'] <= 5].index.tolist())
        df93 = set(data[data['40'] > -25].index.tolist())
        df94 = set(data[data['50'] > 0.5].index.tolist())
        df95 = set(data[data['48'] <= 40].index.tolist())

        df7 = df9 & df91 & df92 & df93 & df94 & df95
        df7 = list(df7)
        df_all = data.ix[df7, [0, 1]]
    elif yinzi=='model2':
        #print(data)
        data_copy = copy.deepcopy(data)
        data_copy = Factor1(data_copy,0,10,6)
        data_copy = Factor2(data_copy, -0.1, 1, 10, 6)
        data_copy = Factor3(data_copy, -0.1, 0.8, 10, 6)
        data_copy = Factor4(data_copy, 0, 10, 6)
        data_copy = Factor5(data_copy, -3, 3, 10, 7)
        data_copy = Factor6(data_copy, 0.15)
        data_copy = Factor7(data_copy, 0.1)
        data_copy = Factor8(data_copy, 1.5, 10, 1)

        df1 = set(data_copy[data_copy['factor1'] ==1].index.tolist())
        df2 = set(data_copy[data_copy['factor2'] == 1].index.tolist())
        df3 = set(data_copy[data_copy['factor3'] == 1].index.tolist())
        df4 = set(data_copy[data_copy['factor4'] == 1].index.tolist())
        df5 = set(data_copy[data_copy['factor5'] == 1].index.tolist())
        df6 = set(data_copy[data_copy['factor6'] == 1].index.tolist())
        df7 = set(data_copy[data_copy['factor7'] == 1].index.tolist())
        df8 = set(data_copy[data_copy['factor8'] == 1].index.tolist())

        df = df1&df2&df3&df4&df5&df6&df7&df8
        a1 = round(len(df1) / len(data_copy),2)
        a2 = round(len(df2) / len(data_copy),2)
        a3 = round(len(df3) / len(data_copy),2)
        a4 = round(len(df4) / len(data_copy),2)
        a5 = round(len(df5) / len(data_copy),2)
        a6 = round(len(df6) / len(data_copy),2)
        a7 = round(len(df7) / len(data_copy),2)
        a8 = round(len(df8) / len(data_copy),2)
        print('factor1',a1)
        print('factor2', a2)
        print('factor3', a3)
        print('factor4', a4)
        print('factor5', a5)
        print('factor6', a6)
        print('factor7', a7)
        print('factor8', a8)

        print ('符合条件的数量，总数量，占比',len(df),len(data_copy),round(len(df)/len(data_copy),2))
        df7 = list(df)

        df7 = [i for i in df7 if i > 0 and i < len(data)]
        df_all = data.ix[df7, [0, 1]]

    elif yinzi == 'MACD':
        df1 = data[data['9'] > 0].reset_index(drop=True).round(4)
        df_all = df1.iloc[::, 0:2]
    elif yinzi[:4] == 'ROCP':
        a = 2
        df111 = set(data[data['40'] > 7].index.tolist())
        df1112 = set(data[data['40'] < 20].index.tolist())
        df222 = set(data[data['48'] > 12].index.tolist())

        df99 = set(data[data['31'] > 3].index.tolist())

        df991 = set([i + 1 for i in df99])
        df992 = set([i + 2 for i in df99])
        df666 = set(data[data['25'] > 1].index.tolist())

        df1 = set(data[data['0'] > 1].index.tolist())
        df12 = set(data[data['1'] > a].index.tolist())
        df13 = set(data[data['31'] > a].index.tolist())
        df14 = set(data[data['3'] > 6].index.tolist())
        df15 = set(data[data['4'] > a].index.tolist())
        df3 = set([i + 1 for i in df1])
        df2 = set([i + 2 for i in df1])

        df32 = set([i + 1 for i in df12])
        df22 = set([i + 2 for i in df12])

        df33 = set([i + 1 for i in df13])
        df23 = set([i + 2 for i in df13])

        df34 = set([i + 1 for i in df14])
        df24 = set([i + 2 for i in df14])

        df35 = set([i + 1 for i in df15])
        df25 = set([i + 2 for i in df15])
        # df77=df1 | df12 | df13 | df14 | df3 | df2 | df32 | df22 | df33 | df23 | df34 | df24 | df15 | df35 | df25| df99|df991|df992
        # df77 = df12 | df13  | df32  | df33  | df15 | df35 | df99 | df34| df14 | df25 |df3|df2|df991|df992 |df1
        df77 = df13 | df33  # &df666  #| df222
        df7 = df77  # -df9 -df_rise -df_rise_down_30-df_down4-df_down3 -df_down5 #df77

        # df7=df7&df98
        # print (len(df77),len(df7),len(df9),len(data))
        df7 = list(df7)

        df7 = [i for i in df7 if i > 0 and i < len(data)]
        df_all = data.ix[df7, [0, 1]]

    elif yinzi == 'OP_CL2':
        df9 = set(data[data['24'] <= 5].index.tolist())
        df91 = set(data[data['24'] >= -9.8].index.tolist())

        df7 = df9 & df91
        df7 = list(df7)
        df_all = data.ix[df7, [0, 1]]

    return df_all

