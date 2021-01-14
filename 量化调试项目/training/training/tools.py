import pandas as pd
import tushare as ts
import time
import datetime
import os
import config2 as cn
import QUANTAXIS as QA

#获取最近的50个交易日的日期和指数涨跌幅
def get_date_range(date,day_long):
    trade_date = QA.trade_date_sse
    trade_date2 = [time.mktime(time.strptime(i,"%Y-%m-%d")) for i in trade_date]
    e_time = time.mktime(time.strptime(date,"%Y-%m-%d"))

    for i in range(len(trade_date2)):
        if i>1:
            if trade_date2[i-1]<=e_time and trade_date2[i]>=e_time:
                abc = i
                break
    #abc = trade_date.index(date) #.index.tolist()[0] #-1
    abc2 = abc-day_long
    date_list= trade_date[abc2:abc]  # 得到日期列表

    # 得到对应日期的上证指数涨跌幅
    data = QA.QA_fetch_index_day_adv('000001', '2013-06-10', date).data.reset_index()
    data['rise_down'] = data['close'].pct_change() * 100
    data = data.round(2).dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    shIndex_chglist = data[data['date'].isin(date_list)]
    return date_list, shIndex_chglist

def update_trade_date():
    # 更新交易日历，每年更新一次
    import QUANTAXIS as QA
    import config as cn
    # data = QA.trade_date_sse
    # print(data)
    pro = ts.pro_api(cn.ts_token)
    data = pro.query('trade_cal', start_date='', end_date='')
    # data = pro.trade_cal(exchange='', start_date='', end_date='')
    data.to_csv(cn.trade_date2, mode='w', index=False, header=True)
    data = data[data['is_open'].isin([1])].reset_index(drop=True)
    data.to_csv(cn.trade_date, mode='w', index=False, header=True)
    print(data)

def get_stock_all2():
    zhongzheng_dir = os.path.join(cn.sim_data_file, 'aaa.csv')
    code = pd.read_table('aaaa.txt', sep=',', dtype=str)['code'].tolist()  # time.strptime(s1,'%Y%m%d')
    stock_list1 = []
    for code in code:

        if code[0] == '6':
            k = 'sh' + code + '.csv'
            stock_list1.append(k)
        else:
            k = 'sz' + code + '.csv'
            stock_list1.append(k)
    return stock_list1

def get_stock_all():
    """
    默认获取全市场的市值前1500的股票列表
    :param sort: sort='mktcap' 按市值排序
    :param len:  返回的列表长度
    :param update： 是否重新生成
    :return:
    """
    import QUANTAXIS as QA
    stock_data = QA.QAFetch.QATdx.QA_fetch_get_stock_list('stock')
    stock_list = stock_data['code'].tolist()
    stock_list1 = []
    for  code in stock_list:

        if code[0] == '6':
            k = 'sh' + code +'.csv'
            stock_list1.append(k)
        else:
            k = 'sz' + code +'.csv'
            stock_list1.append(k)

    return stock_list1


def get_stock_list(sort='mktcap', len=4000, update=False):
    """
    默认获取全市场的市值前1500的股票列表
    :param sort: sort='mktcap' 按市值排序
    :param len:  返回的列表长度
    :param update： 是否重新生成
    :return:
    """
    dir_stock = r"./file/stock_all.txt"
    zhongzheng_dir = os.path.join(cn.sim_data_file, 'stock_all_mktcap.csv')
    print("开始更新股票池列表", update)
    if update:

        import QUANTAXIS as QA
        global QA
        if os.path.exists(zhongzheng_dir):
            os.remove(zhongzheng_dir)
        # 获取股票的最新一期的财务报表数据
        stock_data = QA.QAFetch.QATdx.QA_fetch_get_stock_list('stock')
        #print(stock_data)
        df_st = stock_data.loc[stock_data.name.str.contains('S')]
        stock_name = set(stock_data['name'].tolist()) - set(df_st['name'].tolist())
        stock_data = stock_data[stock_data['name'].isin(stock_name)]
        if True:
            for code in stock_data['code'].tolist():
                try:
                    df = QA.QA_fetch_get_stock_info('tdx',code)
                    if os.path.exists(zhongzheng_dir):
                        header = False
                    else:
                        header = True
                    df.to_csv(zhongzheng_dir, mode='a+', index=False, header=header)
                except Exception as e:
                    print(code, '错误', e)

    data = pd.read_table(zhongzheng_dir, sep=',', dtype=str)  # time.strptime(s1,'%Y%m%d')
    print(data.shape)
    data['ipo_date'] = data['ipo_date'].astype('float64')
    data = data[(data['ipo_date'] > 0)]
    data['ipo_date'] = [str(i)[:4] + '-' + str(i)[4:6] + '-' + str(i)[6:] for i in data['ipo_date'].tolist()]
    # 得到最近50个交易日的日期以及上证涨跌幅列表
    date_list, shIndex_chglist = get_date_range(cn.date, 30)

    stock_list = []
    for index, row in data.iterrows():
        if row['ipo_date'] not in date_list: #上市日期不在最近一年工作日列表内，就不是次新股
            code = str(row['code']).zfill(6)
            if code[0:2] == '60':
                k = 'sh' + code+ '.csv'
                stock_list.append(k)
            elif code[0:3] == '300' or code[0:2] == '00':
                k = 'sz' + code + '.csv'
                stock_list.append(k)

    stock_list = set(stock_list[:len])

    return stock_list




# 获取初始股票池代码列表
def get_all_stockcode():
    stock_list = get_stock_list(sort='mktcap', len=4000, update=cn.update)

    print(len(stock_list))
    data = pd.read_excel(cn.stock_pool_file, sheet_name=None)['Sheet1']
    data.columns = ['排名', '股票代码', '上涨次数', '分组', '是否样本']
    abc = data[data['上涨次数'] >= 9].reset_index(drop=True)['股票代码'].tolist()
    abc = [str + '.csv' for str in abc]
    stock_all =  set(stock_list)&set(abc)
    return stock_all

# 获取初始股票池代码列表
def get_top_longtou():

    data = pd.read_excel(cn.stock_pool_longtou, sheet_name=None)['Sheet1']
    stock_all = [str(i).zfill(6)+ '.csv' for i in data['代码'].tolist()]
    stock_list = []
    for code in stock_all:
        if code[0] == '6':
            k = 'sh' + code
            stock_list.append(k)
        else:
            k = 'sz' + code
            stock_list.append(k)
    return stock_list


def heiming_dan(stock_list):
    data_hm = pd.read_table(cn.stock_hm_file, sep=',')  #黑名单列表
    stock_hm=data_hm['code'].tolist()
    stock_hm=[str(i).zfill(6) for i in stock_hm]
    stock_hm = hzhui(stock_hm)
    abc = set(stock_list) - set(stock_hm)  # -set(stock_hm)
    stock_list = [str + '.csv' for str in abc]
    return stock_list

def heiming_dan2():
    data_zy = pd.read_excel(cn.stock_zy_file, sheet_name=None)['Sheet1']
    d = data_zy.iloc[1].tolist()
    data_zy.columns = d
    data_zy = data_zy.iloc[2:, :]
    data_zy['质押比例（%）'] = data_zy['质押比例（%）'].astype(float)
    abc2 = data_zy[data_zy['质押比例（%）'] >= 50]['证券代码'].tolist()
    abc2 = hzhui(abc2)
    return abc2




def hzhui(abc2):
    n = []
    for i in abc2:
        if i[:2] == '60':
            m = 'sh' + i
            n.append(m)
        elif i[:2] == '00' or i[:2] == '30':
            m = 'sz' + i
            n.append(m)
    return n
# 全市场股票代码
def get_all_stockcode0(sort='mktcap', len=4000, update=False):
    """
       默认获取全市场的市值前1500的股票列表
       :param sort: sort='mktcap' 按市值排序
       :param len:  返回的列表长度
       :param update： 是否重新生成
       :return:
       """
    dir_stock = r"./file/stock_all.txt"
    zhongzheng_dir = os.path.join(cn.sim_data_file, 'stock_all_mktcap.csv')
    print("开始更新股票池列表", update)
    if update:

        import QUANTAXIS as QA
        global QA
        if os.path.exists(zhongzheng_dir):
            os.remove(zhongzheng_dir)
        # 获取股票的最新一期的财务报表数据
        stock_data = QA.QAFetch.QATdx.QA_fetch_get_stock_list('stock')
        # print(stock_data)
        df_st = stock_data.loc[stock_data.name.str.contains('S')]
        stock_name = set(stock_data['name'].tolist()) - set(df_st['name'].tolist())
        stock_data = stock_data[stock_data['name'].isin(stock_name)]
        if True:
            for code in stock_data['code'].tolist():
                try:
                    df = QA.QA_fetch_get_stock_info('tdx', code)
                    if os.path.exists(zhongzheng_dir):
                        header = False
                    else:
                        header = True
                    df.to_csv(zhongzheng_dir, mode='a+', index=False, header=header)
                except Exception as e:
                    print(code, '错误', e)

    data = pd.read_table(zhongzheng_dir, sep=',', dtype=str)  # time.strptime(s1,'%Y%m%d')
    print(data.shape)
    data['ipo_date'] = data['ipo_date'].astype('int')
    data = data[(data['ipo_date'] > 0)]
    data['ipo_date'] = [str(i)[:4] + '-' + str(i)[4:6] + '-' + str(i)[6:] for i in data['ipo_date'].tolist()]
    # 得到最近50个交易日的日期以及上证涨跌幅列表
    date_list, shIndex_chglist = get_date_range(cn.date, 0)
    stock_list = []
    for index, row in data.iterrows():
        # if str(row['code']).zfill(6)=='002980':
        #     print(date_list)
        #     print(row['ipo_date'])
        # if row['ipo_date'] not in date_list: #上市日期不在最近一年工作日列表内，就不是次新股
        code = str(row['code']).zfill(6)
        if code[0:1] == '6':
            k = 'sh' + code + '.csv'
            stock_list.append(k)
        elif code[0:3] == '300' or code[0:2] == '00':
            k = 'sz' + code + '.csv'
            stock_list.append(k)

    stock_list = list(set(stock_list[:len]))

    return stock_list


# 获取沪深300代码列表
def hushen_300():
    a=ts.get_hs300s()['code'].tolist()
    n = []
    for i in a:
        if i[:2] == '60':
            m = 'sh' + i
        elif i[:2] == '00' or i[:2] == '30':
            m = 'sz' + i
        n.append(m)

    n =ST(n)  # 排除ST
    # 获得需要预测的股票
    abc = [str + '.csv' for str in n]
    return abc




# 排除ST的股票
def ST(abc):
    ST_dir = os.path.join(cn.sim_data_file, 'zhongzheng.csv')
    df = pd.read_table(ST_dir, sep=',')
    #print(print(df))
    df['code']=[str(i).zfill(6) for i in df['code'].tolist()]
    df = df.ix[df.name.str.contains('S')]
    #print(df)

    b = [i[2:8] for i in abc]
    a = df['code'].tolist()

    c = set(b) & set(a)
    d = set(b) - c
    n = []

    for i in d:
        if i[:2] == '60':
            m = 'sh' + i
        elif i[:2] == '00' or i[:2] == '30':
            m = 'sz' + i
        n.append(m)
    return n







def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) * childern_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

#得到当日没更新的股票数据
def stock_today(stock_code_list):
    min_data_dir=cn.qa_min_data_dir
    today=str(datetime.date.today())
    #today='2018-12-03'
    day=int(today[-2:])
    month=int(today[5:7])
    year=int(today[:4])
    stock=[]
    for filename in os.listdir(min_data_dir):
        dataset_dir562 = os.path.join(min_data_dir, filename)
        statinfo=os.stat(dataset_dir562)
        b=time.localtime(statinfo.st_mtime) #['tm_year']
        year2,month2,day2=b[0],b[1],b[2]
        if year==year2 and month==month2 and day==day2:
            stock.append(filename)
    stock_code_list=set(stock_code_list)-set(stock)
    return stock_code_list

# 等待....直到指定时间返回
def wait(_hour, _minute):
    curTime = datetime.datetime.now()
    desTime = curTime.replace(hour=_hour, minute=_minute, second=0, microsecond=0)
    delta = desTime-curTime
    sleepSeconds = delta.total_seconds()

    if sleepSeconds > 0:
        time.sleep(sleepSeconds)

    return True

# 每周更新一次基础数据，包括st与股票池
def update_basic_data():
    from datetime import datetime, date
    dayOfWeek = datetime.today().weekday()
    if dayOfWeek >=4:
        cn.update = True  # ST股票池更新为True，不更新为False
        stock_list = get_stock_list(sort='mktcap', len=4000, update=cn.update)


if __name__ == '__main__':
    #update_basic_data()
    update_trade_date()

    #a = get_all_stockcode()
    a = get_stock_list(sort='mktcap', len=4000, update=True)
    #b = get_top_longtou()
    print(len(a))
    # import QUANTAXIS as QA
    # stock_data = QA.QAFetch.QATdx.QA_fetch_get_stock_list('stock')
    # print(len(stock_data))
    #update_trade_date()

#if 'sz300175.csv' in a:
    #print(123)