import os
import datetime


#回测与训练存放的各个参数
base_dir = r'D:\project' #os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

#计算因子的参数-------------------------------------------------------------------------------------------------------------
stock_hm_file = base_dir +  r'\data\heimingdan.csv'	#黑名单股票的目录
stock_file_dir = base_dir +  r'\stock_simulation\Stock_date\stock_file'
stock_pool_file = base_dir +  r'\stock_simulation\Stock_date\stock_file\stock_pool.xlsx'	#初始股票池的目录
sim_data_file = base_dir +  r'\data' #初始目录

qa_his_day_data_dir = base_dir +  r'\data\Market_Data\qa_History_Date'  # 历史日线数据存放目录
if not os.path.exists(qa_his_day_data_dir):
    os.makedirs(qa_his_day_data_dir)

qa_his_min_data_dir = base_dir +  r'\data\Market_Data\qa_History_30Min'  # 历史分钟数据存放目录
if not os.path.exists(qa_his_min_data_dir):
    os.makedirs(qa_his_min_data_dir)

# qa_his_min_data_dir = base_dir +  r'\data\Market_Data\qa_History_30Min'  # 历史分钟数据存放目录
# qa_his_day_data_dir = base_dir +  r'\data\Market_Data\qa_History_Date'  # 历史日线数据存放目录

huice_data_dir = base_dir +  r'\stock_simulation\Stock_date\train_data'
if not os.path.exists(huice_data_dir):
    os.makedirs(huice_data_dir)

huice_data_dir2 = base_dir +  r'\stock_simulation\Stock_date\train_data\subfile_file'
if not os.path.exists(huice_data_dir2):
    os.makedirs(huice_data_dir2)

huice_shIndex_data_dir = base_dir +  r'\data\Market_Data\Market_Data_index.txt'  # 上证指数存放的目录

#huice_shIndex_data_dir = base_dir +  r'\stock_simulation\Stock_date\stock_file\Market_Data_index.txt'  # 上证指数历史数据存放的目录
huice_rise_down_dir=base_dir +  r'\stock_simulation\Stock_date\train_data\zhangdiefu3.csv'
huice_shujuji_dir=base_dir +  r'\stock_simulation\Stock_date\train_data\shujuji.csv'
mor_Stock_price=base_dir +  r'\stock_simulation\Stock_date\train_data\mor_Stock_price.csv'

selector1 = ['min_feature30','min_close_open30','min_low_open30','min_high_open30','min_high_close30','min2_feature30']

selector2 = ['ma']

selector = ["QRR", "OROCP", "ROCP", "HROCP", "LROCP", "MA", "close_ma5",
            "close_ma10", "MACD", "Volatility_ratio", "VROCP", "rise_down", "factor1", "factor2", "factor3",
            "factor4", 'late_Stock_price30', 'late_QRR30', 'mor_Stock_price30', 'mor_QRR30',
            'late_Stock_price60', 'late_QRR60', 'mor2_Stock_price60', 'mor2_QRR60', 'mor_Stock_price60', 'open_close'
    , 'min_close_open', 'min_low_open', 'min_low_close', 'min_high_open', 'min_high_close'
    , 'mor_QRR60', 'mor_hou_Stock_price30', 'mor_hou_QRR30', 'rise_3_days', 'rise_down_30_days',
            'min_close_open2', 'min_low_open2', 'min_low_close2', 'min_high_open2', 'min_high_close2',
            'date_open_close', 'rise_20_days', 'tor', 'osd','rise_down_3_5'] +selector1 + selector2 #, 'hushen_300']

window = 1  # 时间窗口

#模型训练的参数-------------------------------------------------------------------------------------------------------------

shujuji='shujuji.csv'
#读取数据集的目录
shuju_file = base_dir +  r'\stock_simulation\Stock_date\train_data\%s'% (shujuji)

date = str(datetime.date.today())

