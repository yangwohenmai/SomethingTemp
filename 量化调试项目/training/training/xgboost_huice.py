#encoding:utf-8
"""
模型训练程序修改：
1. model_config.py      不同模型读入不同的配置, 将config.py中的模型参数独立出来
2. model_train_data.py  载入模型参数, 根据模型参数生成训练集、验证集
3. model_train.py       载入模型参数, 读入对应的训练集进行训练，读入验证集进行验证，运行回测
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from yinzi import yinzi
from pandas import DataFrame  # DataFrame通常来装二维的表格
import os,re
import config2 as cn
import nitModel as nm
from datetime import datetime
import xgboost as xgb
import warnings
import  tools
import model_config as mc
from qa_export import get_trade_dates
warnings.filterwarnings("ignore")  #忽略警告



def shIndex_data(start,end):
    # data = pd.read_table(cn.huice_shIndex_data_dir, sep=',')
    import QUANTAXIS as QA
    data = QA.QA_fetch_index_day_adv('000001', start, end).data.reset_index()
    data['date'] = data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # print(data)
    # exit()
    # data = data.rename(columns={'datetime': 'date'})
    abc = data.loc[data['date'] == start]['date'].index.tolist()
    abc2 = data.loc[data['date'] == end]['date'].index.tolist()
    if len(abc)==0:
        print ('start is not workday')
        exit()
    if len(abc2) == 0:
        print('end is not workday')
        exit()
    date_list = data.loc[abc[0]:abc2[0], ['date']]['date'].tolist()  # 得到日期列表
    data['rise_down']=data['close'].pct_change() * 100
    data = data.round(2).dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    data = data[data['date'].isin(date_list)]['rise_down'].tolist()  # data
    #date_list = [re.sub('-', '/', i) for i in date_list]
    return date_list,data

# 把辅助因子增加到预测明细中用于多因子的筛选
def zheng_he(df4,trainFileName255):

    factor_else = pd.read_table('./file/factor_else.csv', sep=',')  # 读取训练数据集
    #df4.columns = ['code', 'date', 'rise_down', 'class', 'pro']
    print(len(factor_else.columns))
    factor_else.columns = ['code', 'date','量比', '昨日涨幅','当日早盘涨幅', '当日早盘量比',
                           '最近3交易日涨幅', '最近20交易日涨幅', '昨日交易日前价格', '换手率',
                           '买入当天的10:30价格', '买入当天的收盘价格', '买入第二天的开盘价', '买入第二天10:00的最高价', '买入第二天10:30的最高价', '买入第二天10:30的价格',
                           '过去三天涨幅', '过去五天涨幅','最高价日期','最高价的成交量与前一天的之比', '最高价前5天涨幅', '收盘价/开盘价', '最高价前5天内涨停的个数',
                            'T+1涨幅', 'T+2涨幅','T+3涨幅', 'T+4涨幅', 'T+5涨幅']
    # df4['code']=[i[2:8] for i in df4['code'].tolist()]

    df3 = pd.merge(df4, factor_else, on=['code', 'date'])
    dataset_dir4=os.path.join(trainFileName255, 'test_pre.csv')
    #print (df3)
    df3.to_csv(dataset_dir4, mode='w', index=False, header=True)
    print ('end')


def read_pre(dataset_dir3):


    data_pre = pd.read_table(dataset_dir3, sep=',', header=None)  # 预测明细

    data_pre.columns = ['code', 'date', 'rise_down', 'class', 'pro', 'pro0', 'pro1', 'pro2']

    data_pre = data_pre[data_pre['class'] == 2].reset_index(drop=True)
    # data_pre2 = data_pre[(data_pre['date'] == '2017-12-29')].reset_index(drop=True).sort_values('pro', ascending=False)
    # print(data_pre2)
    # data_pre3 = data_pre.sort_values('pro', ascending=False).groupby('date', group_keys=False).head(10)
    # data_pre3 = data_pre3[~data_pre3['date'].isin(['2017-12-29'])]
    # data_pre3.to_csv(r'./file/Testdata.csv', mode='w', index=False, header=True)

    return data_pre

# 10只股票的平均回测
def back_trader(predict_data_dir,back_data):


    predict_data = read_pre(predict_data_dir) # 读取预测明细，并排除涨停的股票数据

    #zheng_he(predict_data, trainFileName255)  # 输出辅助因子到明细中

    predict_data = predict_data[predict_data['class'] ==2].reset_index(drop=True)
    predict_data['date'] = [datetime.strptime(i, '%Y-%m-%d') for i in predict_data['date'].tolist()]
    start = str(min(predict_data['date'].tolist()))[0:10]
    end = str(max(predict_data['date'].tolist()))[0:10]

    date_list, sh_data = shIndex_data(start, end)
    #print (date_list)

    hui_zong = DataFrame(index=['下跌', '上涨<1', '上涨>1', '准确率>0', '准确率>1', '指数收益率', '指数涨跌净值','平均收益率','模型净值'])
    m,n,n9,hh=1,1,1,0

    for y in date_list:
        #print(y)
        predict_data_day = predict_data[predict_data['date'].isin([y])]
        predict_data_day=predict_data_day.sort_values(by='pro', ascending=False).reset_index(drop=True)
        # if  y=='2015-05-28':
        #     print(66,y,predict_data_day)
        #     dataset_dir337 = os.path.join(trainFileName255, 'f'+y[-2:]+'ttt.csv')
        #     predict_data_day.to_csv(dataset_dir337, mode='w', index=False, header=True)

        num1,num2,num3=0,0,0
        rise_down_list=[]
        for each_stock in predict_data_day.iterrows():
            each_stock = each_stock[1]
            if len(rise_down_list) > 9:
                break
            rise_down = each_stock['rise_down']
            rise_down_list.append(rise_down)
            if rise_down <= 0:
                num1 = num1+1          #下跌
            if 0 < rise_down < 1:
                num2 = num2+1         #上涨小于1
            if rise_down >= 1:
                num3 = num3+1          #上涨大于1

        if (num1 + num2 + num3) == 0:
            zhunque_lv1=0
        else:
            zhunque_lv1=(num2+num3)/(num1+num2+num3)   #准确率1

        if (num1+num2 + num3)==0:
            zhunque_lv2=0
        else:
            zhunque_lv2 = (num3) / (num1+num2 + num3)   #准确率2

        if (num1+num2 + num3) == 0:
            aveg1 =0
        else:
            aveg1=sum(rise_down_list)/(num1+num2 + num3) -0.15    #平均收益率

        m = m+m*(aveg1/100*(num1+num2 + num3)/10)                        #模型净值
        m29 = 0 #sh_data[hh]  # 指数收益率
        n9 =0 # n9 * (1 + m29 / 100)  # 指数涨跌净值

        zhunque_lv1 = '{0:.2f}%'.format(zhunque_lv1 * 100)
        zhunque_lv2 = '{0:.2f}%'.format(zhunque_lv2 * 100)

        d=[num1,num2,num3,zhunque_lv1,zhunque_lv2,m29,n9,aveg1,m]
        #print (d)
        hui_zong[y]=d

        hh=hh+1
    hui_zong=hui_zong.T
    #print (huizong.columns)
    hui_zong=hui_zong[['下跌', '上涨<1', '上涨>1', '准确率>0', '准确率>1', '指数收益率', '指数涨跌净值','平均收益率','模型净值']]

    hui_zong.round(3).to_csv(back_data, mode='w', index=True, header=True)

 #对每只股票训练完的参数进行一个总的汇总，保存到E:\new file\lstm\xgboost_all\..\result目录下
 #股票单独训练或者，合并的数据集训练都得用到这个函数，进行一个汇总



def load_data(trainFileName, ff, timestep, kk, data33, ww, csvfile):
    # 读入csv数据

    data=trainFileName
    print (data.shape,1)

    data2 = data.round(2).values
    q = data2.shape[1]-18

    t = [i+1 for i in range(18) if i+1!=kk]
    tt = [q+i-1 for i in t]
    data_all = np.delete(data2, tt, axis=1)
    data = load_data_factor(data_all, ff, timestep)

    print(data.shape, 2)
    #factor_else(data2, timestep) #输出明细中增加辅助多因子

    data = DataFrame(data).dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) #.values
    if ww:
        print (len(data),666)
        data=load_data_yinzi(data,data33)
        print(len(data), 777)
    else:
        data =data.values
    print(data.shape, 3)

    df3 = DataFrame(data)
    df3.round(2).to_csv(csvfile, mode='w', index=False, header=True)

    return data

def load_data_yinzi(data,data33):
    data33['hebing'] = data33['daima'] + data33['data']
    data['hebing'] = data[0] + data[1]
    a = set(data33['hebing'].tolist())
    a1 = set(data['hebing'].tolist())
    a3 = set.intersection(a1, a)
    data = data[data['hebing'].isin(a3)]  # 训练集
    del data['hebing']
    data = data.values
    return data


def factor_else(data,timestep):
    print(data)
    ff=[0,2,30,37,40,48,49,50,52,53,54,55,56,57,58,59,65,66,67,68,69,-5,-4,-3,-2,-1]
    data0 = data[timestep - 1:, 0:2]
    data1 = data[timestep - 1:, 2:]
    print (data.shape,data1.shape)
    data2 = data1[::, ff]
    data4 = np.hstack((data0, data2))
    df3 = DataFrame(data4)
    df3.round(2).to_csv('./file/factor_else.csv', mode='w', index=False, header=True)
    print ('end')



def load_data_factor(data,ff,timestep):
    data0 = data[timestep - 1:, 0:2]
    data1 = data[::, 2:-1]
    data2 = data1[::, ff]
    data3 = data[timestep-1:, -1]
    data3 = data3[:, np.newaxis]

    yy=[]
    for i in range(data2.shape[0]-timestep+1):
        data6 = data2[i:i+timestep,::]
        data6 = np.reshape(data6, (1, -1))
        yy.append(data6)

    data9 = np.reshape(np.array(yy), (data2.shape[0]-timestep+1, -1))

    data4 = np.hstack((data0, data9))
    data5 = np.concatenate((data4, data3), axis=1)



    #df34 = DataFrame(data)
    #df34.round(2).to_csv('./test2.csv', mode='w', index=False, header=True)

    #df3 = DataFrame(data5)
    #df3.round(2).to_csv('./test.csv', mode='w', index=False, header=True)

    return data5


def get_trade_days(start, end):
    """
    从指数文件中获取start~end之间的交易日列表
    :param start:
    :param end:
    :return:
    """
    data = pd.read_table(cn.huice_shIndex_data_dir, sep=',')
    data = data.rename(columns={'datetime': 'date'})
    print(data)


    abc = data.loc[data['date'] == start]['date'].index.tolist()
    abc2 = data.loc[data['date'] == end]['date'].index.tolist()
    if len(abc)==0:
        print ('start is not workday')
        exit()
    if len(abc2) == 0:
        print('end is not workday')
        exit()

    date_list=data.loc[abc[0]:abc2[0], ['date']]['date'].tolist()  # 得到日期列表

    return date_list


def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) *childern_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list


def train_and_test(train_dataset, test_dataset):
    """
    模型训练、测试
    :param train_dataset:   训练集
    :param test_dataset:    测试集(样本外)
    :return:
    """
    global kk                            #指定标签列 1~5
    if train_model:
        # train_csvfile = os.path.join(trainFileName255, 'train.csv')
        # dataset = load_data(trainFileName, factors, timestep, kk, data33,ww, train_csvfile)

        print('训练集', train_dataset.shape)
        X = train_dataset[:, 2:-1]
        Y = train_dataset[:, -1]
        nm.lable_Y(Y, num_class)

        seed = 4
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_val = xgb.DMatrix(X_test, label=y_test)

        # 训练模型
        model = nm.get_xgb_model(xgb_train, xgb_val, num_class)
        print("best best_ntree_limit", model.best_ntree_limit)
        d = model.best_ntree_limit
        with open(trainFileName_model2, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            f.write(str(d))
        model.save_model(trainFileName_model)

        # 验证集上预测
        y_pred = model.predict(xgb_val, ntree_limit=model.best_ntree_limit) # model.best_ntree_limit

        b=[]
        b1=[]
        for c in y_pred.tolist():
            b2=c.index(max(c))
            b.append(b2)
            b1.append(max(c))

        predictions = [int(value) for value in b]

        # 结果评估
        accuracy = accuracy_score(y_test.tolist(), predictions)
        print("TrainDataSet Accuracy: %.2f%%" % (accuracy * 100.0))

        true_num, pred_num, pred_accuracy = nm.class_accuracy_score(y_test.tolist(), predictions, num_class)
        print("TrainDataSet true_num:" , true_num)
        print("TrainDataSet true_ratio:" , [str(round(item*100.0/len(y_test),2))+'%' for item in true_num])
        print("TrainDataSet pred_num:" , pred_num)
        print("TrainDataSet pred_accuracy:" , pred_accuracy)

        #绘制特征的重要度
        importance = model.get_score(importance_type='weight')
        #importances = model.feature_importances_
        #print ('importances',importance)
        importance5 = [0 for i in range(num_factor)]
        print(importance5)
        for key in importance:
            i=int(key[1::])
            h=i%num_factor
            #print(i, h,importance5[h])
            importance5[h]=importance5[h]+importance[key]   #各个特征重要度的相加
        factor_list=['factor'+str(i+1) for i in range(num_factor)]
        importance2 = pd.DataFrame(index=['aaa'],columns=(factor_list))
        importance2.loc['aaa'] = importance5     #特征的汇总
        contents=model.best_ntree_limit
    else:
        #实盘10点的模型
        # testFileName_model=r'E:\new file\lstm\simulation\data\fangzhen\all_file\1000_1000_2011-04-01-2019-01-24_timestep3_label2_factors19\model.xgb'
        # testFileName_model2=r'E:\new file\lstm\simulation\data\fangzhen\all_file\1000_1000_2011-04-01-2019-01-24_timestep3_label2_factors19\best_ntree_limit.txt'

        # 实盘10点半的模型,52
        testFileName_model = r'E:\new file\lstm\simulation\data\fangzhen\all_file\1000_455_2012-01-10-2017-12-28_timestep3_label1_factors27\model.xgb'
        testFileName_model2 = r'E:\new file\lstm\simulation\data\fangzhen\all_file\1000_455_2012-01-10-2017-12-28_timestep3_label1_factors27\best_ntree_limit.txt'

        model = xgb.Booster(model_file=testFileName_model)
        with open(testFileName_model2, 'r') as f:  # 默认模式为‘r’，只读模式
            contents = int(f.read())  # 读取文件全部内容

    # 样本外测试
    # test_csvfile = os.path.join(trainFileName255, 'test.csv')
    # test_dataset = load_data(testFileName, factors, timestep, kk, data44,ww2, test_csvfile)

    print('测试集', test_dataset.shape)
    name = test_dataset[:,0]
    data = test_dataset[:,1]
    X_test = test_dataset[:,2:-1]
    y_test = test_dataset[:,-1]

    xgb_test = xgb.DMatrix(X_test)
    y_pred = model.predict(xgb_test, ntree_limit=contents)
    b = []
    b1 = []
    for c in y_pred.tolist():
        b2 = c.index(max(c))
        b.append(b2)
        b1.append(max(c))
    #print (66,y_pred)

    df4 = DataFrame()
    df4['name']=name
    df4['data']=data
    df4['True']=y_test.tolist()
    df4['Predict']=b
    df4['Predict2'] = b1
    df4['Predict-1'] = y_pred[:,0]
    df4['Predict0'] = y_pred[:,1]
    df4['Predict1'] = y_pred[:,2]

    #factor_else = pd.read_table('./factor_else.csv', sep=',')  # 读取训练数据集

    nm.lable_Y(y_test, num_class)
    predictions = [int(value) for value in b]
    accuracy = accuracy_score(y_test.tolist(), predictions, normalize=True)
    print("TestDataSet Accuracy: %.2f%%" % (accuracy * 100.0))

    true_num, pred_num, pred_accuracy = nm.class_accuracy_score(y_test.tolist(), predictions, num_class)
    print("TestDataSet true_num:" , true_num)
    print("TestDataSet true_ratio:" , [str(round(item*100.0/len(y_test),2))+'%' for item in true_num])
    print("TestDataSet pred_num:" , pred_num)
    print("TestDataSet pred_accuracy:" , pred_accuracy)

    j=['label','factor','num_class','class_block','timestep','true_ratio','pred_accuracy','true_num','pred_num','promotion']
    kk3=[str(round(item * 100.0 / len(y_test), 2)) + '%' for item in true_num]
    jj=[kk,str(factors),num_class,'<=-1,(-1,1),>=1',timestep,str(kk3)
        ,str(pred_accuracy),str(true_num),str(pred_num),float(pred_accuracy[2].strip('%'))-float(kk3[2].strip('%'))]

    df5 = DataFrame(index=['aaa'],columns=(j))
    df5.loc['aaa'] = jj
    #df5 = pd.concat([df5, importance2], axis=1)

    if os.path.exists(testFileName_predict):
        header = False
    else:
        header = True
    #if float(pred_accuracy[2].strip('%'))-float(kk3[2].strip('%'))>5:
    df5.to_csv(testFileName_predict, mode='w', index=False, header=header)
    if creat_pre == True:
        df4.round(5).to_csv(trainFileName2, mode='w', index=False, header=False)  #用于存储样本外的预测明细

def read_stock_data():
    #stock_list = tools.get_stock_list(sort='mktcap', len=1500, update=False)
    #stock_list = tools.hushen_300()
    #print("训练股票范围：", len(stock_list))
    stock_list = tools.get_stock_list() #get_stock_all()
    #stock_list = tools.get_all_stockcode()
    stock_list = [str[:-4] for str in stock_list]
    #print(stock_list)
    if 'sh600666' in stock_list:
        print("训练股票范围：", len(stock_list))
    else:
        print(123)

    data = pd.read_table(shuju_file, sep=',')  # 读取训练数据集
    print(data.columns)

    data = data[data['daima'].isin(stock_list)].reset_index(drop=True)
    # a = data['data'].value_counts()
    # print(a)
    # exit()

    # data2 = data.iloc[:1000,:]
    # data2.to_csv('123.csv', mode='w', index=False, header=True)

    return data

def model_train():
    global num_class,shujuji,today,shuju_file,testFileName_predict,trainFileName2,\
    trainFileName_model,trainFileName_model2,creat_pre,factors,timestep,num_factor,train_model,kk
    # 加载10:00模型的参数配置
    #mc.load_config_1000()

    # 加载10:30模型的参数配置
    # mc.load_config_1030()

    # 加载10:00模型回测的参数配置
    # mc.load_config_huice_1000()

    # 加载10:30模型回测的参数配置
    mc.load_config_huice_1030()

    # 加载10:30模型回测的参数配置
    # mc.load_config_huice_1030()
    # 加载趋势模型的参数配置
    #mc.load_config_trend()

    factors_all=cn.factors_all
    kk_all=cn.kk_all       #标签天数
    num_class = cn.num_class
    timestep_all = cn.timestep_all
    creat_pre = cn.creat_pre           #输出预测明细为True，否则为False
    train_model=cn.train_model
    shujuji=cn.shujuji      #选择训练和测试数据集
    ww = cn.ww        #训练集是否需要做筛选
    ww2 =cn.ww2                #测试集是否需要做筛选
    start=cn.start         #训练集与测试集的总时间段
    end=cn.end
    h2=cn.h2                  #训练集的天数
    ratio=cn.ratio           #滚动时测试集的天数
    h3=ratio                    #测试集的天数
    all_times=h2+ratio          #每一次训练与预测时，训练集与测试集总的天数

    #缠论标签化'label_hua1','label_hua3','label_hua4','label_hua1_3','label_hua2_4'
    yinzi2=cn.yinzi2         #数据集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2' 训练集的筛选 'OP_CL'
    yinzi3=cn.yinzi3         #数据集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2'  测试集的筛选

    ww3=cn.ww3                 #是否需要筛选掉不需要的时间段
    start2=cn.start2         #删除不需要的时间段`
    end2=cn.end2
    today=cn.today          #合并的数据中滚动训练的文件的保存日期
    shuju_file=cn.shuju_file #读取数据集的目录
    trainFileName255 = cn.trainFileName255 # 循环存汇总结果
    trainFileName25 = cn.trainFileName25 # 循环存明细结果
    print(trainFileName255)


    if not os.path.exists(trainFileName25):
        os.makedirs(trainFileName25)
    if not os.path.exists(trainFileName255):
        os.makedirs(trainFileName255)

    #进行滚动训练
    train_interval = get_trade_dates(start, end) #get_trade_days(start, end)
    #train_interval, shIndex_chglist = get_date_range(end, 600)#得到需要回测的总时间段

    print ('总时间段', len(train_interval),train_interval[0],train_interval[-1])
    if ww3:
        f2=get_trade_days(start2, end2)                  #得到不需要的时间段
    else:
        f2=[]
    train_interval=[i for i in train_interval if i not in f2]                 #得到需要的时间段

    list_of = list_of_groups(train_interval[h2::], h3)            #滚动训练时的滚动次数，对列表进行等分
    print ('滚动次数',len(list_of))

    # 读入shujuji
    if cn.load_model_dataset is False:
        data_shujuji = read_stock_data()

        data3 = yinzi(yinzi2, data_shujuji)     #计算训练数据集筛选的因子
        data4 = yinzi(yinzi3, data_shujuji)     #计算测试数据集筛选的因子

    for m in range(len(list_of)):
        print ('_________________________________________________滚动训练开始：',m)
        m2= train_interval[int(m * h3):int(m * h3) + all_times]
        m3=m2[0:h2]
        m4=m2[h2::]
        print (len(m2),len(m3),len(m4))

        if cn.load_model_dataset is False:
            train_shujuji = data_shujuji[data_shujuji['data'].isin(m3)]         # 训练集
            test_shujuji = data_shujuji[data_shujuji['data'].isin(m4)]          # 测试集

            data33 = data3[data3['data'].isin(m2)]      # 训练数据集因子的筛选
            data44 = data4[data4['data'].isin(m2)]      # 测试数据集因子的筛选

        mm2 = "%s_%s_%s-%s_result.csv" % (h2,h3,re.sub('/', '-', start),re.sub('/', '-', end))
        testFileName_predict = os.path.join(trainFileName25, mm2)       # 测试集 re.sub('/', '-', start)

        for tstep in timestep_all:
            for kk1 in kk_all:
                for factors in factors_all:
                    print('______________________________________________________________________', m, tstep)

                    num_factor = len(factors)
                    timestep = tstep
                    kk=kk1

                    #mm = "%s_%s_%s-%s_timestep%s_label%s_factors%s" % (h2,h3,re.sub('/', '-', start),re.sub('/', '-', end), timestep,kk,num_factor)
                    train_result_dir=trainFileName25 #os.path.join(trainFileName25, mm)
                    trainFileName2 = os.path.join(train_result_dir, 'TestDataSet_predict.csv')   # 保存测试集结果 xgb.model
                    trainFileName_model = os.path.join(train_result_dir, 'model.xgb')             # 保存测试集结果 xgb.model  'model.xgb'
                    trainFileName_model2=os.path.join(train_result_dir, 'best_ntree_limit.txt')             # 保存测试集结果 xgb.model  'model.xgb'

                    # 创建每个参数的目录
                    if creat_pre == True:
                        if not os.path.exists(train_result_dir):
                            os.makedirs(train_result_dir)

                    # 加载或生成模型训练集和测试集
                    train_csvfile = os.path.join(trainFileName255, 'train.csv')

                    if train_model:
                        if cn.load_model_dataset :
                            train_dataset = pd.read_table(train_csvfile, sep=',').values
                        else:
                            train_dataset = load_data(train_shujuji, factors, timestep, kk, data33, ww, train_csvfile)
                    else:
                        train_dataset = 0

                    test_csvfile = os.path.join(trainFileName255, 'test.csv')
                    if cn.load_model_dataset:
                        test_dataset = pd.read_table(test_csvfile, sep=',').values
                    else:
                        test_dataset = load_data(test_shujuji, factors, timestep, kk, data44, ww2, test_csvfile)

                    # 训练并测试模型
                    train_and_test(train_dataset, test_dataset)

    #进行总的汇总，保存到E:\new file\lstm\xgboost_all\..\result目录下

    #回测程序，对每个参数进行回测并保存到汇总的文件中，文件名按照timestep的参数来命名
    for root, dirs, files in os.walk(trainFileName25, topdown=True):
        for i in dirs:

            dataset_dir39 = os.path.join(trainFileName25, i)
            dataset_dir3 = os.path.join(dataset_dir39, 'TestDataSet_predict.csv')
            timestep = i.split('_')[3]
            dataset_dir337 = os.path.join(trainFileName255, timestep + 'huice2.csv')
            #nq.end_chuli(dataset_dir3, dataset_dir337, trainFileName255)
            print(dataset_dir3,dataset_dir337,trainFileName255)
            back_trader(dataset_dir3,dataset_dir337,trainFileName255)

if __name__ == '__main__':
    # 测试集预测的结果保存目录
    aa = r'D:\project\stock_simulation\Stock_date\stock_result\all_file\2020-05-02-1030_ROCP_4_shujuji._wwTrue_ww2True_100_2000_label3_calss3_factor27'
    dataset_dir3 = aa + r'\TestDataSet_predict.csv'

    # 回测结果保存目录
    back_data = r'xgboost_huice_1030.csv'
    back_trader(dataset_dir3, back_data)