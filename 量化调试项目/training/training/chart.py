# coding:UTF-8
# Copyright 2017 The Xiaoyu Fang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy
import talib
import math
import pandas as pd
import matplotlib
#matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from pandas import DataFrame  # DataFrame通常来装二维的表格  factor4 alpha001
import random
import re
import time as tm
import numpy as np
import tushare as ts
import copy,os
from datetime import datetime
from datetime import timedelta



class ChartFeature(object):
    def __init__(self, selector):
        self.selector = selector
        self.supported = {'ma','min_feature30','min_close_open30','min_low_open30','min_high_open30','min_high_close30','min2_feature30','rise_down_3_5','open_close2','vr','rise_20_days','tor','osd','hushen_300','rise_20_price','date_open_close','min_close_open2', 'min_low_open2', 'min_low_close2', 'min_high_open2', 'min_high_close2','rise_3_days','rise_down_30_days','mor_QRR60','mor_hou_Stock_price30', 'mor_hou_QRR30','min_close_open', 'min_low_open', 'min_low_close', 'min_high_open', 'min_high_close','open_close','xia_ying', 'mor2_Stock_price30', 'mor2_QRR30','label_hua','mor2_Stock_price60','mor2_QRR60','late_Stock_price60','late_QRR60','mor_Stock_price60','mor_QRR60','late_Stock_price30','late_QRR30','mor_Stock_price30','mor_QRR30','day_Stock_price','day_QRR',"theory","ROCP","QRR","rise_down","Stock_price", "OROCP", "HROCP", "LROCP", "MACD", "RSI",
                          "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME", "close_down", "close_ma5", "close_ma10",
                          "Volatility_ratio","factor1","factor2","factor3","factor4",'alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026', 'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha048', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha056', 'alpha057', 'alpha058', 'alpha059', 'alpha060', 'alpha061', 'alpha062', 'alpha063', 'alpha064', 'alpha065', 'alpha066', 'alpha067', 'alpha068', 'alpha069', 'alpha070', 'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha076', 'alpha077', 'alpha078', 'alpha079', 'alpha080', 'alpha081', 'alpha082', 'alpha083', 'alpha084', 'alpha085', 'alpha086', 'alpha087', 'alpha088', 'alpha089', 'alpha090', 'alpha091', 'alpha092', 'alpha093', 'alpha094', 'alpha095', 'alpha096', 'alpha097', 'alpha098', 'alpha099', 'alpha100', 'alpha101', 'alpha102', 'alpha103', 'alpha104', 'alpha105', 'alpha106', 'alpha107', 'alpha108', 'alpha109', 'alpha110', 'alpha111', 'alpha112', 'alpha113', 'alpha114', 'alpha115', 'alpha116', 'alpha117', 'alpha118', 'alpha119', 'alpha120', 'alpha121', 'alpha122', 'alpha123', 'alpha124', 'alpha125', 'alpha126', 'alpha127', 'alpha128', 'alpha129', 'alpha130', 'alpha131', 'alpha132', 'alpha133', 'alpha134', 'alpha135', 'alpha136', 'alpha137', 'alpha138', 'alpha139', 'alpha140', 'alpha141', 'alpha142', 'alpha143', 'alpha144', 'alpha145', 'alpha146', 'alpha147', 'alpha148', 'alpha149', 'alpha150', 'alpha151', 'alpha152', 'alpha153', 'alpha154', 'alpha155', 'alpha156', 'alpha157', 'alpha158', 'alpha159', 'alpha160', 'alpha161', 'alpha162', 'alpha163', 'alpha164', 'alpha165', 'alpha166', 'alpha167', 'alpha168', 'alpha169', 'alpha170', 'alpha171', 'alpha172', 'alpha173', 'alpha174', 'alpha175', 'alpha176', 'alpha177', 'alpha178', 'alpha179', 'alpha180', 'alpha181', 'alpha182', 'alpha183', 'alpha184', 'alpha185', 'alpha186', 'alpha187', 'alpha188', 'alpha189', 'alpha190', 'alpha191'}
        self.feature = []

    def moving_extract(self, window=30, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                       volumes=None, with_label=True, flatten=True,data=None,daima=None,rise_down=None,rise_down2=None,theory=None,data2=None,xulie=None,filename=None,raw_data=None,label_hua=None,jiben_mian=None,hushen_300_code_list=None,tor=None,vr=None):
        self.extract(open_prices=open_prices, close_prices=close_prices, high_prices=high_prices, low_prices=low_prices,
                     volumes=volumes,data=data,daima=daima,rise_down2=rise_down2,theory=theory,data2=data2,xulie=xulie,filename=filename,raw_data=raw_data,label_hua=label_hua,jiben_mian=jiben_mian,hushen_300_code_list=hushen_300_code_list,tor=tor,vr=vr)
        feature_arr = numpy.asarray(self.feature)
        p = 0
        rows = feature_arr.shape[0]
        #print(feature_arr)
        #print("feature dimension: %s" % rows)
        #print ('feature_arr.shape[1]',feature_arr.shape)

        if with_label:
            moving_features = []
            moving_labels = []

            x2 = feature_arr[:2,  window::]
            moving_labels = rise_down[window::,::]
            while p + window < feature_arr.shape[1]:
                x = feature_arr[2:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                #moving_labels.append(y)
                p += 1


            return numpy.asarray(moving_features), numpy.asarray(moving_labels),x2
        else:
            moving_features = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                p += 1
            return moving_features

    def extract(self, open_prices=None, close_prices=None, high_prices=None, low_prices=None, volumes=None,data=None,daima=None,rise_down2=None,theory=None,data2=None,xulie=None,filename=None,raw_data=None,label_hua=None,jiben_mian=None,hushen_300_code_list=None,tor=None,vr=None):
        self.feature = []
        self.feature.append(daima)
        self.feature.append(data)

        for feature_type in self.selector:
            if feature_type in self.supported:
                #print("extracting feature : %s" % feature_type)
                self.extract_by_type(feature_type, open_prices=open_prices, close_prices=close_prices,
                                     high_prices=high_prices, low_prices=low_prices, volumes=volumes,rise_down2=rise_down2,theory=theory,data=data,xulie=xulie,filename=filename,raw_data=raw_data,jiben_mian=jiben_mian,hushen_300_code_list=hushen_300_code_list,tor=tor,vr=vr)
            else:
                print("feature type not supported: %s" % feature_type)
        #self.feature_distribution()
        return self.feature

    def feature_distribution(self):
        k = 0
        for feature_column in self.feature:
            fc = numpy.nan_to_num(feature_column)
            mean = numpy.mean(fc)
            var = numpy.var(fc)
            max_value = numpy.max(fc)
            min_value = numpy.min(fc)
            print("[%s_th feature] mean: %s, var: %s, max: %s, min: %s" % (k, mean, var, max_value, min_value))
            k = k + 1

    def extract_by_type(self, feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                        volumes=None,rise_down2=None,theory=None,data=None,xulie=None,filename=None,raw_data=None,jiben_mian=None,hushen_300_code_list=None,tor=None,vr=None):

        if feature_type == 'ma':
            rise_down_1 = (close_prices[1:] / close_prices[:-1] - 1) * 100
            rise_down_1 = [0 for i in range(1)] + rise_down_1.tolist() #5
            ma5 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=5))
            ma10 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=10))
            #self.feature.append(ma5)
            #self.feature.append(ma10)
            ma5_101 = [ ma5[i] - ma10[i]  for i in range(len(ma5))] #1

            ma5_rise_down = (ma5[1:] / ma5[:-1] - 1) * 100
            ma5_rise_down = [0 for i in range(1)] + ma5_rise_down.tolist() #2


            ma10_rise_down = (ma10[1:] / ma10[:-1] - 1) * 100
            ma10_rise_down = [0 for i in range(1)] + ma10_rise_down.tolist() #3


            ma10_4 = [close_prices[i] - ma10[i]  for i in range(len(ma10))] #4


            high_low_6 = [max(high_prices[i-9:i+1])/min(low_prices[i-9:i+1])-1 if i>=10 else 0 for i in
                              range(len(close_prices))]  #6
            #self.feature.append(high_low_6)
            close_low_7 = [close_prices[i] / min(low_prices[i - 4:i + 1]) - 1 if i >= 5 else 0 for i in
                          range(len(close_prices))]  #7

            self.feature.append(ma5_101)  # MA5>MA10，10日内出现的次数>=7
            self.feature.append(ma5_rise_down)  # 0<(MA5 - 昨日MA5)*100/昨日MA5<0.6，10日内出现的次数>=7
            self.feature.append(ma10_rise_down)  # 0<(MA10 - 昨日MA10)*100/昨日MA10<0.4，10日内出现的次数>=7
            self.feature.append(ma10_4)  # CLOSE>MA10，10日内出现的次数>=6
            self.feature.append(rise_down_1)  # -3%<收盘涨幅<3%，10日内出现的次数>=7
            self.feature.append(high_low_6)  # (近10日内最高价-近10日最低价)/近10日最低价<15%
            self.feature.append(close_low_7)  # (CLOSE-近5日最低价)/近5日最低价<10%

            qrr = np.array([volumes[i] / volumes[i - 4:i+1].mean() if i > 4 else 0 for i in range(len(volumes))])
            #self.feature.append(qrr)
            factor8 = []
            for i in range(len(ma5_101)):

                if i >= 10:
                    l8 = max(qrr[i - 9:i + 1])
                else:
                    l8 = 0
                factor8.append(l8)
            self.feature.append(factor8)  # 10日内出现一次量比>=2 [1:].tolist() +[0]

            factor9 = raw_data['ma5_10_00'].tolist()[1:]+[0]
            factor10 = raw_data['ma5_10_30'].tolist()[1:]+[0]
            factor13 = raw_data['ma2_5_10_00'].tolist()[1:] + [0]
            factor14 = raw_data['ma2_5_10_30'].tolist()[1:] + [0]

            factor11 = raw_data['price_10_00'].tolist()[1:]+[0]
            factor12 = raw_data['price_10_30'].tolist()[1:]+[0]
            self.feature.append(factor9)  # 10日内出现一次量比>=2
            self.feature.append(factor10)  # 10日内出现一次量比>=2
            self.feature.append(factor13)  # 10日内出现一次量比>=2
            self.feature.append(factor14)  # 10日内出现一次量比>=2

            self.feature.append(factor11)  # 10日内出现一次量比>=2
            self.feature.append(factor12)  # 10日内出现一次量比>=2
            self.feature.append(close_prices)  # 收盘价
            self.feature.append(open_prices)  # 开盘价
            self.feature.append(low_prices)  # 最低价
            self.feature.append(high_prices)  # 最高价
            self.feature.append(amount)  # 成交额




        if feature_type == 'sz_factor':  # [1:].tolist() +[0]
            rise_down_1 = (close_prices[1:] / close_prices[:-1] - 1) * 100  # T-1日上证涨幅
            rise_down_1 = rise_down_1.tolist()
            b = [0 for i in range(1)]
            rise_down_1 = b + rise_down_1
            self.feature.append(rise_down_1)  # T-1日涨幅

            close_ma5 = []
            ma5 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=5))
            for i in range(len(ma5)):
                # print (close_prices,ma5)
                if i >= 10:
                    if close_prices[i] > ma5[i]:
                        l = 1
                    else:
                        l = 0
                else:
                    l = 0
                close_ma5.append(l)
            self.feature.append(close_ma5)   # if T-1收盘价>MA5，1,2

            close_ma10 = []
            ma10 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=10))
            for i in range(len(ma10)):
                if i >= 10:
                    if close_prices[i] > ma10[i]:
                        l = 1
                    else:
                        l = 0
                else:
                    l = 0
                close_ma10.append(l)
            self.feature.append(close_ma10)   # if T-1日收盘价<MA10,1,2

            close_ma_down5 = []
            close_ma_down10 = []
            for i in range(len(ma10)):
                if i >= 5:
                    l3 = sum(close_ma5[i-4:i+1])
                    l4 = sum(close_ma10[i - 4:i + 1])
                    if l3 >= 1:
                        l = 1
                    else:
                        l = 0

                    if l4 >= 1:
                        l2 = 1
                    else:
                        l2 = 0

                else:
                    l = 0
                    l2 = 0
                close_ma_down5.append(l)
                close_ma_down10.append(l2)

            self.feature.append(close_ma_down5)   # 近5日上证是否跌破MA5,1,2
            self.feature.append(close_ma_down10) # 近5日上证是否跌破MA10,1,2

            close_ma30 = []
            ma30 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=30))
            for i in range(len(ma10)):
                if close_prices[i] > ma30[i]:
                    l = 1
                else:
                    l = 2
                close_ma30.append(l)
            self.feature.append(close_ma30)

            close_ma120 = []
            ma120 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=120))
            for i in range(len(ma10)):
                if close_prices[i] > ma120[i]:
                    l = 1
                else:
                    l = 2
                close_ma120.append(l)
            self.feature.append(close_ma120)

            signal, hist, macd = myMACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            self.feature.append(macd.tolist()) # T-1日收盘价MACD


            macd_1 = []
            macd_2 = []
            for i in range(len(macd)):

                if i >= 5:
                    macd_low5 = min(macd[i-4:i+1])
                    a=macd[i-4:i+1]
                    l2 = len(np.where((a<0))[0])
                    if macd_low5 == macd[i]:
                        l = 1
                    else:
                        l = 0
                else:
                    l = 0
                    l2 = 0
                macd_1.append(l)
                macd_2.append(l2)
            self.feature.append(macd_1) # T-1日MACD是否5日新低
            self.feature.append(macd_2) # 5日内MACD<0的天数






        #'min_feature30', 'min_close_open30', 'min_low_open30', 'min_high_open30', 'min_high_close30', 'min2_feature30'
        min_num_15 = ['min' + str(i) for i in range(8)]
        time_min_15_1 = [i + '30分钟涨跌幅' for i in min_num_15]
        time_min_15_2 = [i + '30分钟量比' for i in min_num_15]
        time_min_15_3 = [i + 'min_close_open' for i in min_num_15]
        time_min_15_4 = [i + 'min_low_open' for i in min_num_15]
        time_min_15_5 = [i + 'min_high_open' for i in min_num_15]
        time_min_15_6 = [i + 'min_high_close' for i in min_num_15]
        time_min_15_7 = [i + '30分钟涨跌幅2' for i in min_num_15]
        time_min_15_8 = [i + '30分钟量比2' for i in min_num_15]

        if feature_type == 'min_feature30': # [1:].tolist() +[0] 'ma'
            for name in time_min_15_1:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)
            for name in time_min_15_2:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)
        if feature_type == 'min_close_open30':
            for name in time_min_15_3:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)
        if feature_type == 'min_low_open30':
            for name in time_min_15_4:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)
        if feature_type == 'min_high_open30':
            for name in time_min_15_5:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)
        if feature_type == 'min_high_close30':
            for name in time_min_15_6:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)
        if feature_type == 'min2_feature30':
            for name in time_min_15_7:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)
            for name in time_min_15_8:
                feature = raw_data[name][1:].tolist() +[0]
                self.feature.append(feature)

        if feature_type == 'xia_ying':
            xia_ying = close_prices/low_prices
            self.feature.append(xia_ying)
        #'min_close_open', 'min_low_open', 'min_low_close', 'min_high_open', 'min_high_close' 'rise_down_3_5'
        #'mor_QRR60','mor_hou_Stock_price30', 'mor_hou_QRR30'
        #'mor_Stock_price60','mor_QRR60','mor_hou_Stock_price30', 'mor_hou_QRR30'
        #'min_close_open2', 'min_low_open2', 'min_low_close2', 'min_high_open2', 'min_high_close2',
        #'open_close2'
        #'rise_20_days','tor','osd','hushen_300','rise_20_price'

        if feature_type == 'rise_down_3_5':
            rise_down_3 = (close_prices[3:] / close_prices[:-3] - 1) * 100 # 过去3日涨幅
            rise_down_3 = rise_down_3.tolist()
            b = [0 for i in range(3)]
            rise_down_3 = b + rise_down_3
            self.feature.append(rise_down_3)

            rise_down_5 = (close_prices[5:] / close_prices[:-5] - 1) * 100 # 过去5日涨幅
            rise_down_5 = rise_down_5.tolist()
            b = [0 for i in range(5)]
            rise_down_5 = b + rise_down_5
            self.feature.append(rise_down_5)

            rise_down2_1 = (close_prices[1:] / close_prices[:-1] - 1) * 100 # 未来1日涨幅
            rise_down2_1 = rise_down2_1.tolist()
            b = [0 for i in range(1)]
            rise_down2_1 = rise_down2_1 + b
            self.feature.append(rise_down2_1)

            rise_down2_2 = (close_prices[2:] / close_prices[:-2] - 1) * 100 # 未来2日涨幅
            rise_down2_2 = rise_down2_2.tolist()
            b = [0 for i in range(2)]
            rise_down2_2 = rise_down2_2 + b
            self.feature.append(rise_down2_2)

            rise_down2_3 = (close_prices[3:] / close_prices[:-3] - 1) * 100  # 未来3日涨幅
            rise_down2_3 = rise_down2_3.tolist()
            b = [0 for i in range(3)]
            rise_down2_3 = rise_down2_3 + b
            self.feature.append(rise_down2_3)

            rise_down2_4 = (close_prices[4:] / close_prices[:-4] - 1) * 100 # 未来4日涨幅
            rise_down2_4 = rise_down2_4.tolist()
            b = [0 for i in range(4)]
            rise_down2_4 = rise_down2_4 + b
            self.feature.append(rise_down2_4)

            rise_down2_5 = (close_prices[5:] / close_prices[:-5] - 1) * 100   # 未来五日涨幅
            rise_down2_5 = rise_down2_5.tolist()
            b = [0 for i in range(5)]
            rise_down2_5 = rise_down2_5 + b
            self.feature.append(rise_down2_5)

            b,b2,b4,b5 = [],[],[],[]
            high_prices2 =list(high_prices)
            for indexs in range(len(high_prices2)):  # 获取五日内最高价日期
                day1,day2 = 3,5
                if indexs >= day1:
                    # print (indexs)
                    high_index = high_prices2[indexs-day1+1:indexs+1].index(max(high_prices2[indexs-day1+1:indexs+1]))
                    c = data[indexs-day1 + high_index]
                    d = volumes[indexs-day1 + high_index]/volumes[indexs-day1-1 + high_index]
                    e = (close_prices[indexs-day1 + high_index]/open_prices[indexs-day1 + high_index]- 1) * 100
                    if indexs >= day2:
                        price = high_prices2[indexs - day1 + high_index]
                        price_10= price/close_prices[indexs - day1-day2 + high_index]
                    else:
                        price_10 = 0

                else:
                    c = 0
                    d = 0
                    price_10 = 0
                    e = 0
                b.append(c)
                b2.append(d)
                b4.append(price_10)
                b5.append(e)
            self.feature.append(b)           # 获取五日内最高价日期
            self.feature.append(b2)          # 最高价的成交量与前一天的之比
            self.feature.append(b4)           # 离最高价的5天涨幅
            self.feature.append(b5)           # 收盘价/开盘价

            #lrocp = (close_prices / open_prices - 1) * 100   # 收盘价/开盘价
            #self.feature.append(lrocp)

            lrocp = (close_prices[1:] / close_prices[:-1] - 1) * 100  # 涨跌幅
            lrocp = lrocp.tolist()
            lrocp.insert(0, 0)
            lrocp = numpy.asarray(lrocp)
            b3 = []
            for indexs in range(len(lrocp)):                # 10天内涨停的个数
                if indexs > 9:
                    num_9_5 = len(lrocp[indexs-10:indexs][lrocp[indexs-10:indexs] > 9.5])
                else:
                    num_9_5 = 0
                b3.append(num_9_5)

            self.feature.append(b3)








        if feature_type == 'open_close2':
            #Volatility_ratio = (close_prices - open_prices) / (high_prices - low_prices)

            mor_QRR = raw_data.values[::, 25][1:].tolist() +[0]
            colse_1 = raw_data.values[::, 28][1:].tolist() +[0]
            open_2  = raw_data.values[::, 29][2:].tolist() +[0,0]

            mor_QRR2 = raw_data.values[::, 26][2:].tolist()+[0,0]

            mor_QRR3 = raw_data.values[::, 27][2:].tolist()+[0,0]
            mor_QRR4 = raw_data.values[::, 25][2:].tolist() + [0,0]

            self.feature.append(mor_QRR)
            self.feature.append(colse_1)
            self.feature.append(open_2)
            self.feature.append(mor_QRR2)
            self.feature.append(mor_QRR3)
            self.feature.append(mor_QRR4)

        if feature_type == 'rise_20_days':
            rise_20_days = (close_prices[20:] / close_prices[:-20] - 1) * 100
            rise_20_days = rise_20_days.tolist()
            b = [0 for i in range(20)]
            rise_20_days=b+rise_20_days
            rise_20_prices=b+close_prices[20:].tolist()

            self.feature.append(rise_20_days)
            self.feature.append(rise_20_prices)

        if feature_type == 'tor':
            self.feature.append(tor)

        if feature_type == 'vr':
            self.feature.append(vr)

        if feature_type == 'osd':
            if False:
                outstanding = jiben_mian.loc[filename[2:8], 'outstanding']
                outstanding2=numpy.asarray([outstanding]*len(close_prices))
                price = close_prices
                FAMC = outstanding2 * price  # 流通市值计算
            FAMC = numpy.asarray([0]*len(close_prices))
            self.feature.append(FAMC)


        if feature_type == 'hushen_300':
            if filename[2:8] in hushen_300_code_list:
                hushen=True
            else:
                hushen =False

            hushen1=[hushen]*len(close_prices)
            self.feature.append(hushen1)


        if feature_type == 'min_close_open2':
            mor_QRR = raw_data.values[::,19][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_low_open2':
            mor_QRR = raw_data.values[::,20][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_low_close2':
            mor_QRR = raw_data.values[::,21][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_high_open2':
            mor_QRR = raw_data.values[::,22][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_high_close2':
            mor_QRR = raw_data.values[::,23][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)

        if feature_type == 'date_open_close':
            mor_QRR = raw_data.values[::,24][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)




        if feature_type == 'rise_down_30_days':
            rise_down_30_days = (close_prices[1:] / close_prices[:-1] - 1) * 100
            rise_down_30_days = rise_down_30_days.tolist()
            rise_down_30_days.insert(0, 0)
            rise_down_30_days = np.array(rise_down_30_days)
            # print (rise_down_30_days)
            rise_down_30_days[rise_down_30_days < -9] = -10
            e = []
            for i in range(len(close_prices)):
                if i > 29:
                    b = rise_down_30_days[i - 30:i]
                    c = str(b.tolist()).count("-10")
                    if c >= 5:
                        d = 0
                    else:
                        d = 1
                else:
                    d = 1
                e.append(d)

            # print (len(e),len(close_prices),e)

            self.feature.append(e)

        if feature_type == 'rise_3_days':
            rise_3_days = (close_prices[3:] / close_prices[:-3] - 1) * 100
            rise_3_days = rise_3_days.tolist()
            rise_3_days.insert(0, 0)
            rise_3_days.insert(0, 0)
            rise_3_days.insert(0, 0)
            self.feature.append(rise_3_days)






        if feature_type == 'mor_QRR60':
            mor_QRR = raw_data.values[::,16][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'mor_hou_Stock_price30':
            mor_QRR = raw_data.values[::,17][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'mor_hou_QRR30':
            mor_QRR = raw_data.values[::,18][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)


        if feature_type == 'min_close_open':
            mor_QRR = raw_data.values[::,11][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_low_open':
            mor_QRR = raw_data.values[::,12][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_low_close':
            mor_QRR = raw_data.values[::,13][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_high_open':
            mor_QRR = raw_data.values[::,14][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'min_high_close':
            mor_QRR = raw_data.values[::,15][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)




        if feature_type == 'mor_QRR30':
            mor_QRR = raw_data.values[::,5][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'mor_Stock_price30':
            mor_Stock_price = raw_data.values[::,4][1:].tolist()
            mor_Stock_price.append(0)
            self.feature.append(mor_Stock_price)
        if feature_type == 'late_QRR30':
            late_QRR = raw_data.values[::,3][1:].tolist()
            late_QRR.append(0)
            self.feature.append(late_QRR)

        if feature_type == 'late_Stock_price30':
            late_Stock_price = raw_data.values[::,2][1:].tolist()
            late_Stock_price.append(0)
            self.feature.append(late_Stock_price)

        if feature_type == 'late_QRR60':
            late_QRR = raw_data.values[::,7][1:].tolist()
            late_QRR.append(0)
            self.feature.append(late_QRR)

        if feature_type == 'late_Stock_price60':
            late_Stock_price = raw_data.values[::,6][1:].tolist()
            late_Stock_price.append(0)
            self.feature.append(late_Stock_price)
        if feature_type == 'mor2_QRR60':
            mor_QRR = raw_data.values[::, 9][1:].tolist()
            mor_QRR.append(0)
            self.feature.append(mor_QRR)
        if feature_type == 'mor2_Stock_price60':
            mor_Stock_price = raw_data.values[::, 8][1:].tolist()
            mor_Stock_price.append(0)
            self.feature.append(mor_Stock_price)

        if feature_type == 'mor_Stock_price60':
            mor_Stock_price = raw_data.values[::, 10][1:].tolist()
            mor_Stock_price.append(0)
            self.feature.append(mor_Stock_price)






        if feature_type == 'theory':
            theory = theory
            self.feature.append(theory)

        if feature_type == 'close_ma5':
            close_ma5=[]
            ma5 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=5))
            #print (len(ma5),ma5)
            #print(len(ma10))
            #print(len(close_prices),close_prices)
            for i in range(len(ma5)):
                #print (close_prices,ma5)
                if close_prices[i]>ma5[i]:
                    l=1
                else:
                    l = 2
                close_ma5.append(l)
            #print (len(close_ma5),close_ma5)
            self.feature.append(close_ma5)

        if feature_type == 'close_ma10':
            close_ma10=[]

            ma10 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=10))
            #print (len(ma5),ma5)
            #print(len(ma10))
            #print(len(close_prices),close_prices)
            for i in range(len(ma10)):
                if close_prices[i]>ma10[i]:
                    l=1
                else:
                    l = 2
                close_ma10.append(l)
            #print (len(close_ma10),close_ma10)
            self.feature.append(close_ma10)

        if feature_type == 'QRR':
            b = []
            for indexs in range(len(volumes)):
                if indexs>4:
                    #print (indexs)
                    abb = volumes[indexs - 5:indexs].mean()
                    c = volumes[indexs] / abb
                else:
                    c=0
                b.append(c)
            self.feature.append(b)

        if feature_type == 'rise_down':
            rise_down2 = rise_down2
            self.feature.append(rise_down2)

        if feature_type == 'ROCP':
            rocp = talib.ROCP(close_prices, timeperiod=1) * 100
            self.feature.append(rocp)

        if feature_type == 'Volatility_ratio':
            #Volatility_ratio =[]
            #for i in range(len(close_prices)):
                #print (close_prices[i],open_prices[i],high_prices[i],low_prices[i])
                #k=(close_prices[i]-open_prices[i])/(high_prices[i]-low_prices[i])
                #Volatility_ratio.append(k)
            Volatility_ratio=(close_prices-open_prices)/(high_prices-low_prices)

            #print ('Volatility_ratio',len(Volatility_ratio),Volatility_ratio)

            self.feature.append(Volatility_ratio)

        if feature_type == 'close_down':
            hrocp = (high_prices[1:] / close_prices[:-1] - 1) * 100
            rocp = (close_prices[1:] / close_prices[:-1] - 1) * 100
            close_down=hrocp-rocp

            close_down = close_down.tolist()
            close_down.insert(0, 0)
            close_down = numpy.asarray(close_down)

            self.feature.append(close_down)


        if feature_type == 'OROCP':
            #orocp = talib.ROCP(open_prices, timeperiod=1)*100
            orocp = (open_prices[1:]/close_prices[:-1]-1) * 100
            orocp=orocp.tolist()
            orocp.insert(0, 0)
            orocp=numpy.asarray(orocp)
            #print (orocp)
            self.feature.append(orocp)
        if feature_type == 'HROCP':
            #hrocp = talib.ROCP(high_prices, timeperiod=1)*100
            hrocp = (high_prices[1:] / close_prices[:-1] - 1) * 100
            hrocp = hrocp.tolist()
            hrocp.insert(0, 0)
            hrocp = numpy.asarray(hrocp)

            self.feature.append(hrocp)

        if feature_type == 'open_close':
            #lrocp = talib.ROCP(low_prices, timeperiod=1)*100
            lrocp = (close_prices / open_prices - 1) * 100
            self.feature.append(lrocp)

        if feature_type == 'LROCP':
            #lrocp = talib.ROCP(low_prices, timeperiod=1)*100
            lrocp = (low_prices[1:] / close_prices[:-1] - 1) * 100
            lrocp = lrocp.tolist()
            lrocp.insert(0, 0)
            lrocp = numpy.asarray(lrocp)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            #signal, hist,macd = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            signal, hist, macd = myMACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)

            norm_signal = numpy.minimum(numpy.maximum(numpy.nan_to_num(signal), -1), 1)
            norm_hist = numpy.minimum(numpy.maximum(numpy.nan_to_num(hist), -1), 1)
            norm_macd = numpy.minimum(numpy.maximum(numpy.nan_to_num(macd), -1), 1)

            zero = numpy.asarray([0])
            macdrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(macd)))), -1), 1)
            signalrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(signal)))), -1), 1)
            histrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(hist)))), -1), 1)
            macd_0= []
            macd_1 = []
            for i in range(len(macd)):
                if macd[i]>0:
                    m=1
                else:
                    m=2
                macd_0.append(m)

            for i in range(len(macd)-1):
                if macd[i+1]>macd[i]:
                    m1=1
                else:
                    m1=2
                macd_1.append(m1)
            macd_1.insert(0, 0)
            macd_1 = numpy.asarray(macd_1)


            self.feature.append(macd)
            self.feature.append(signal)
            self.feature.append(hist)
            #print (len(macdrocp),macdrocp)

            #self.feature.append(macdrocp)
            #self.feature.append(signalrocp)
            #self.feature.append(histrocp)
            self.feature.append(macd_0)
            self.feature.append(macd_1)

        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            #rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            #rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            #rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            self.feature.append(rsi6 / 100.0 - 0.5)
            self.feature.append(rsi12 / 100.0 - 0.5)
            self.feature.append(rsi24 / 100.0 - 0.5)
            # self.feature.append(numpy.maximum(rsi6 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi12 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi24 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            #self.feature.append(rsi6rocp)
            #self.feature.append(rsi12rocp)
            #self.feature.append(rsi24rocp)
        if feature_type == 'VROCP':
            #vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            vrocp = (volumes[1:] / volumes[:-1] )
            vrocp = vrocp.tolist()
            vrocp.insert(0, 0)
            vrocp = numpy.asarray(vrocp)

            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            # self.feature.append(norm_volumes)
            self.feature.append(vrocp)
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)
        if feature_type == 'MA':
            ma5 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=50))
            ma10 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=70))
            ma20 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=20))
            ma30 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=30))

            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)

            #self.feature.append(ma5)
            #self.feature.append(ma10)
            #self.feature.append(ma20)
            #self.feature.append(ma30)

            #self.feature.append(ma5rocp)
            #self.feature.append(ma10rocp)
            #self.feature.append(ma20rocp)
            #self.feature.append(ma30rocp)
            self.feature.append(ma5)
            self.feature.append(ma10)

            #self.feature.append((ma5 - close_prices) / close_prices)
            #self.feature.append((ma10 - close_prices) / close_prices)
            #self.feature.append((ma20 - close_prices) / close_prices)
            #self.feature.append((ma30 - close_prices) / close_prices)

        if feature_type == 'VMA':
            ma5 = talib.MA(volumes, timeperiod=5)
            ma10 = talib.MA(volumes, timeperiod=10)
            ma20 = talib.MA(volumes, timeperiod=20)
            ma30 = talib.MA(volumes, timeperiod=30)


            ma5rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma5, timeperiod=1)))
            ma10rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma10, timeperiod=1)))
            ma20rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma20, timeperiod=1)))
            ma30rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma30, timeperiod=1)))

            #ma180rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma180, timeperiod=1)))
            #ma360rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma360, timeperiod=1)))
            #ma720rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma720, timeperiod=1)))
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)

            #self.feature.append(ma180rocp)
            #self.feature.append(ma360rocp)
            #self.feature.append(ma720rocp)
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma5 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma10 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma20 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma30 - volumes) / (volumes + 1))))

        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv)

        if feature_type == 'factor1':               #量价背离
            factor1 = []
            for i in range(len(close_prices)):
                if i < 25:
                    aa1= 0
                else:
                    aa1 = -np.corrcoef(high_prices[i-25:i]/low_prices[i-25:i], volumes[i-25:i])[1,0]
                factor1.append(aa1)

            #print(len(high_prices))
            #print (len(factor1),factor1)

            self.feature.append(factor1)

        if feature_type == 'factor2':     #CR
            factor2= []
            mid = (high_prices + low_prices) / 2
            #print (mid)

            for i in range(len(close_prices)):
                if i < 25:
                    aa1 = 0
                else:

                    vrocp1 = high_prices[i-24:i] - mid[i-25:i-1]
                    #print(vrocp1)
                    vrocp1[vrocp1 < 0] = 0
                    vrocp2 = mid[i-25:i-1] - low_prices[i-24:i]
                    vrocp2[vrocp2 < 0] = 0
                    aa1=np.sum(vrocp1)/np.sum(vrocp2)*100
                    #print (aa1,np.sum(vrocp1),np.sum(vrocp2))
                factor2.append(aa1)
            self.feature.append(factor2)

        if feature_type == 'factor3':  # PVT

            vrocp1 = (close_prices[1::] - close_prices[:-1])/close_prices[:-1]*volumes[1::]
            #print (vrocp1)

            factor3 =np.cumsum(vrocp1)
            factor3 = factor3.tolist()
            factor3.insert(0, 0)


            #print(len(high_prices))
            #print (len(factor3),factor3)
            self.feature.append(factor3)

        if feature_type == 'factor4':  # AROON上升与下降
            cc,cc2=[],[]
            for i in range(len(close_prices)):
                if i<25:
                    b=0
                    b9=0
                else:
                    b1=high_prices[i-25:i]
                    b2 = low_prices[i - 25:i]
                    b1=b1.tolist()
                    b2 = b2.tolist()
                    vrocp1 = b1.index(max(b1))
                    vrocp2 = b2.index(max(b2))
                    b=(25-vrocp1)/25*100
                    b9 = (25 - vrocp2) / 25 * 100
                cc.append(b)
                cc2.append(b9)
            self.feature.append(cc)
            self.feature.append(cc2)

        if feature_type == 'alpha001':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            #print (3333,len(data7),len(close_prices))
            if len(data7)==len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha002':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha003':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha004':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha005':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha006':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha007':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha008':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha009':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha010':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha011':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha012':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha013':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha014':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)



        if feature_type == 'alpha015':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha016':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha017':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha018':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha019':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha020':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha021':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha022':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)
        if feature_type == 'alpha023':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha024':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha025':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha026':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha027':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha028':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha029':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha030':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha031':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha032':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha033':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha034':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha035':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha036':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha037':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha038':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha039':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha040':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha041':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha042':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)



        if feature_type == 'alpha043':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha044':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha045':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha046':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha047':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha048':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha049':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha050':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha051':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha052':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha053':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha054':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha055':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha056':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)



        if feature_type == 'alpha057':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha058':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha059':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha060':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha061':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha062':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha063':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha064':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha065':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha066':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha067':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha068':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha069':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha070':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)





        if feature_type == 'alpha071':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha072':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha073':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha074':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha075':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha076':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha077':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha078':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha079':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha080':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha081':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha082':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha083':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)


        if feature_type == 'alpha084':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha085':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha086':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha087':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha088':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha089':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha090':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha091':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha092':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha093':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha094':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha095':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha096':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha097':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)



        if feature_type == 'alpha098':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha099':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha100':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha101':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha102':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha103':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha104':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha105':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha106':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha107':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha108':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha109':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha110':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha111':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha112':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha113':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha114':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha115':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha116':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha117':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha118':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha119':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha120':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha121':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha122':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha123':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha124':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)


        if feature_type == 'alpha125':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha126':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha127':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha128':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha129':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha130':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha131':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha132':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha133':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha134':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha135':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha136':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha137':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha138':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha139':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha140':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha141':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha142':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha143':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha144':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha145':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha146':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha147':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha148':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha149':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha150':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha151':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha152':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha153':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha154':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha155':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha156':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha157':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha158':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha159':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha160':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha161':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha162':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha163':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha164':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha165':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha166':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha167':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha168':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha169':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha170':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha171':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha172':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha173':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha174':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha175':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha176':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha177':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha178':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha179':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)


        if feature_type == 'alpha180':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha181':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha182':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha183':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha184':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha185':
            # (((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?-1:(SUM(CLOSE,2)/2<(SUM(CLOSE,8)/8-STD(CLOSE,8))?1:(1<=(VOLUME/MEAN(VOLUME,20))?1:-1))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha186':
            # -1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha187':
            # -1*RANK(SIGN(DELTA(OPEN*0.85+HIGH*0.15,4)))
            # 注:取值排序有随机性
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha188':
            # (RANK(MAX(VWAP-CLOSE,3))+RANK(MIN(VWAP-CLOSE,3)))*RANK(DELTA(VOLUME,3))
            # 感觉MAX应该为TSMAX
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha189':
            # (-1*CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha190':
            # -1*delta(((close-low)-(high-close))/(high-low),1)
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)

        if feature_type == 'alpha191':
            # -1*SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
            # 这里SUM应该为TSSUM
            data7 = data1[data1['index'].isin(xulie)][feature_type].tolist()
            # print (3333,len(data7),len(close_prices))
            if len(data7) == len(close_prices):
                self.feature.append(data7)




def extract_feature(raw_data,raw_data2,raw_data3, selector,xulie,filename, window=30, with_label=True, flatten=True):
    global data1,amount
    #print (raw_data)

    chart_feature = ChartFeature(selector)
    #jiben_mian = ts.get_stock_basics()  # 基本面数据获取
    jiben_mian = 0
    hushen_300_code_list = 0 #ts.get_hs300s()['code'].tolist() amount


    amount = numpy.asarray(raw_data2['amount'].tolist())
    tor = numpy.asarray(raw_data2['tor'].tolist())
    vr = 0 #numpy.asarray(raw_data2['vr'].tolist())

    closes = numpy.asarray(raw_data2['close'].tolist())
    opens = numpy.asarray(raw_data2['open'].tolist())
    highs = numpy.asarray(raw_data2['high'].tolist())
    lows = numpy.asarray(raw_data2['low'].tolist())
    volumes = numpy.asarray(raw_data2['vol'].tolist()) #, dtype='f8'
    data = numpy.asarray(raw_data['日期'].tolist())
    daima = numpy.asarray(raw_data['代码'].tolist())

    rise_down = raw_data.values[::,-18::]
    rise_down2 = numpy.asarray(raw_data3['rise_down'].tolist())
    theory=[]
    #kl=[int(i) for i in raw_data4['type4'].tolist()]
    #volumes = numpy.asarray(real_data, dtype='f8')
    #print (type(volumes[0]),volumes[0])

    if with_label:
        moving_features, moving_labels,x2 = chart_feature.moving_extract(window=window, open_prices=opens,
                                                                      close_prices=closes,
                                                                      high_prices=highs, low_prices=lows,
                                                                      volumes=volumes, with_label=with_label,
                                                                      flatten=flatten,data=data,daima=daima,
                                                                         rise_down=rise_down
                                                                         ,rise_down2=rise_down2,theory=theory,data2=raw_data2,
                                                                         xulie=xulie,filename=filename,raw_data=raw_data,jiben_mian=jiben_mian,hushen_300_code_list=hushen_300_code_list,tor=tor,vr=vr)
        return moving_features, moving_labels,x2
    else:
        moving_features = chart_feature.moving_extract(window=window, open_prices=opens, close_prices=closes,
                                                       high_prices=highs, low_prices=lows, volumes=volumes,
                                                       with_label=with_label, flatten=flatten)
        return moving_features

def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r

def myMACD(price, fastperiod=12, slowperiod=26, signalperiod=9):
    ewma12 = pd.Series(price).ewm(span=fastperiod).mean()
    ewma60 = pd.Series(price).ewm(span=slowperiod).mean()
    #ewma12 = pd.ewma(price,span=fastperiod)
    #ewma60 = pd.ewma(price,span=slowperiod)
    #print (ewma12)

    dif = ewma12-ewma60
    #dea = pd.ewma(dif,span=signalperiod)
    dea = pd.Series(dif).ewm(span=signalperiod).mean()
    bar = (dif-dea) #有些地方的bar = (dif-dea)*2，但是talib中MACD的计算是bar = (dif-dea)*1
    return dif,dea,bar