#encoding:utf-8
import config2 as cn

base_dir = r'D:\project' 

def load_config_1000():
    """
    设置10:00模型的训练参数
    :return:
    """
    factors_all1 = [0, 1, 2, 3, 4, 7, 8, 13, 16, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36]  # , 38,39]  # ,39,42,43,44,45]
    cn.factors_all = [factors_all1]

    cn.kk_all=[2]                       # 标签频率为9:30,10:00,10:30,11:00,14:30
    cn.num_class = 3
    cn.timestep_all = [3]
    cn.creat_pre = True                 #输出预测明细为True，否则为False
    cn.train_model = True

    cn.shujuji='shujuji.csv'            #选择训练和测试数据集
    cn.load_model_dataset = True       #true: 直接从csv文件加载模型的训练集和测试集;  false: 用shujuji生成模型的训练集和测试集

    cn.ww  = True                       #训练集是否需要做筛选
    cn.ww2 = False                      #测试集是否需要做筛选
    cn.start = '2011-04-01'             #训练集与测试集的总时间段
    cn.end   = '2019-01-24'

    cn.ww3=False                        #是否需要筛选掉不需要的时间段
    cn.start2='2018-08-06'              #删除不需要的时间段`
    cn.end2='2016-01-04'

    cn.h2=1000                          #训练集的天数
    cn.ratio=904                       #滚动时测试集的天数
    cn.h3 = cn.ratio                    #测试集的天数
    cn.all_times = cn.h2 + cn.ratio     #每一次训练与预测时，训练集与测试集总的天数

    cn.yinzi2='ROCP_4'              #训练集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2' 训练集的筛选 'OP_CL'
    cn.yinzi3='OP_CL'               #测试集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2'  测试集的筛选

    cn.today='2019-05-30_1000'           #合并的数据中滚动训练的文件的保存日期

    cn.trainFileName255 = base_dir +  r'\stock_simulation\Stock_date\stock_result\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                          % (cn.today,cn.yinzi2,cn.shujuji[0:8],cn.ww,cn.ww2,cn.h2,cn.h3,cn.kk_all[0],cn.num_class,len(cn.factors_all[0]))
    cn.trainFileName25 = base_dir +  r'\stock_simulation\Stock_date\stock_result\all_file\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                         % (cn.today, cn.yinzi2, cn.shujuji[0:8], cn.ww, cn.ww2, cn.h2, cn.h3, cn.kk_all[0], cn.num_class, len(cn.factors_all[0]))

def load_config_1030():
    """
    设置10:30模型的训练参数
    :return:
    """
    factors_all1 = [0, 1, 2, 3, 4, 7, 8, 13, 16, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    cn.factors_all = [factors_all1]

    cn.kk_all=[3]                       # 标签频率为9:30,10:00,10:30,11:00,14:30
    cn.num_class = 3
    cn.timestep_all = [3]
    cn.creat_pre = True                 #输出预测明细为True，否则为False
    cn.train_model = True

    cn.shujuji='shujuji.csv'            #选择训练和测试数据集
    cn.load_model_dataset = False       #true: 直接从csv文件加载模型的训练集和测试集;  false: 用shujuji生成模型的训练集和测试集

    cn.ww  = True                       #训练集是否需要做筛选
    cn.ww2 = False                      #测试集是否需要做筛选
    cn.start = '2012-01-10'             #训练集与测试集的总时间段
    cn.end   = '2017-12-28'

    cn.ww3=False                        #是否需要筛选掉不需要的时间段
    cn.start2='2018-08-06'              #删除不需要的时间段`
    cn.end2='2016-01-04'

    cn.h2=1000                          #训练集的天数
    cn.ratio=453                        #滚动时测试集的天数
    cn.h3 = cn.ratio                    #测试集的天数
    cn.all_times = cn.h2 + cn.ratio     #每一次训练与预测时，训练集与测试集总的天数

    cn.yinzi2='ROCP_4'              #训练集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2' 训练集的筛选 'OP_CL'
    cn.yinzi3='OP_CL'               #测试集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2'  测试集的筛选

    cn.today='2019-04-28-1030'           #合并的数据中滚动训练的文件的保存日期

    cn.trainFileName255 = base_dir +  r'\stock_simulation\Stock_date\stock_result\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                          % (cn.today,cn.yinzi2,cn.shujuji[0:8],cn.ww,cn.ww2,cn.h2,cn.h3,cn.kk_all[0],cn.num_class,len(cn.factors_all[0]))
    cn.trainFileName25 = base_dir +  r'\stock_simulation\Stock_date\stock_result\all_file\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                         % (cn.today, cn.yinzi2, cn.shujuji[0:8], cn.ww, cn.ww2, cn.h2, cn.h3, cn.kk_all[0], cn.num_class, len(cn.factors_all[0]))

def load_config_trend():
    """
    趋势模型的参数配置
    :return:
    """
    factors_all1 = [0, 1, 2, 3, 4, 7, 8, 13, 16, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36]

    cn.factors_all = [factors_all1]

    cn.kk_all=[2]                       # 标签频率为9:30,10:00,10:30,11:00,14:30
    cn.num_class = 3
    cn.timestep_all = [3]
    cn.creat_pre = True                 #输出预测明细为True，否则为False
    cn.train_model = False

    cn.shujuji='shujuji.csv'            #选择训练和测试数据集
    cn.load_model_dataset = False       #true: 直接从csv文件加载模型的训练集和测试集;  false: 用shujuji生成模型的训练集和测试集

    cn.ww  = True                       #训练集是否需要做筛选
    cn.ww2 = False                       #测试集是否需要做筛选
    cn.start = '2016-02-02'             #训练集与测试集的总时间段
    cn.end   = '2019-04-25'

    cn.ww3=False                        #是否需要筛选掉不需要的时间段
    cn.start2='2018-08-06'              #删除不需要的时间段`
    cn.end2='2016-01-04'

    cn.h2=0                          #训练集的天数
    cn.ratio=1000                       #滚动时测试集的天数
    cn.h3 = cn.ratio                    #测试集的天数
    cn.all_times = cn.h2 + cn.ratio     #每一次训练与预测时，训练集与测试集总的天数

    cn.yinzi2='model2'              #训练集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2' 训练集的筛选 'OP_CL'
    cn.yinzi3='model2'               #测试集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2'  测试集的筛选

    cn.today='2019-05-30-model'           #合并的数据中滚动训练的文件的保存日期

    cn.trainFileName255 = base_dir +  r'\stock_simulation\Stock_date\stock_result\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                          % (cn.today,cn.yinzi2,cn.shujuji[0:8],cn.ww,cn.ww2,cn.h2,cn.h3,cn.kk_all[0],cn.num_class,len(cn.factors_all[0]))
    cn.trainFileName25 = base_dir +  r'\stock_simulation\Stock_date\stock_result\all_file\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                         % (cn.today, cn.yinzi2, cn.shujuji[0:8], cn.ww, cn.ww2, cn.h2, cn.h3, cn.kk_all[0], cn.num_class, len(cn.factors_all[0]))



def load_config_huice_1000():
    """
    设置10:00模型的回测参数
    :return:
    """
    factors_all1 = [0, 1, 2, 3, 4, 7, 8, 13, 16, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36]  # , 38,39]  # ,39,42,43,44,45]
    cn.factors_all = [factors_all1]

    cn.kk_all=[2]                       # 标签频率为9:30,10:00,10:30,11:00,14:30
    cn.num_class = 3
    cn.timestep_all = [3]
    cn.creat_pre = True                 #输出预测明细为True，否则为False
    cn.train_model = False

    cn.shujuji='shujuji.csv'            #选择训练和测试数据集
    cn.load_model_dataset = False       #true: 直接从csv文件加载模型的训练集和测试集;  false: 用shujuji生成模型的训练集和测试集

    cn.ww  = True                       #训练集是否需要做筛选
    cn.ww2 = False                      #测试集是否需要做筛选
    cn.start = '2016-03-09'             #训练集与测试集的总时间段
    cn.end   = '2020-03-26'

    cn.ww3=False                        #是否需要筛选掉不需要的时间段
    cn.start2='2018-08-06'              #删除不需要的时间段`
    cn.end2='2016-01-04'

    cn.h2=0                          #训练集的天数
    cn.ratio=10000                      #滚动时测试集的天数
    cn.h3 = cn.ratio                    #测试集的天数
    cn.all_times = cn.h2 + cn.ratio     #每一次训练与预测时，训练集与测试集总的天数

    cn.yinzi2='ROCP_4'              #训练集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2' 训练集的筛选 'OP_CL'
    cn.yinzi3='OP_CL2'               #测试集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2'  测试集的筛选

    cn.today='2020-04-12_1000'           #合并的数据中滚动训练的文件的保存日期

    cn.trainFileName255 = base_dir +  r'\stock_simulation\Stock_date\stock_result\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                          % (cn.today,cn.yinzi2,cn.shujuji[0:8],cn.ww,cn.ww2,cn.h2,cn.h3,cn.kk_all[0],cn.num_class,len(cn.factors_all[0]))
    cn.trainFileName25 = base_dir +  r'\stock_simulation\Stock_date\stock_result\all_file\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                         % (cn.today, cn.yinzi2, cn.shujuji[0:8], cn.ww, cn.ww2, cn.h2, cn.h3, cn.kk_all[0], cn.num_class, len(cn.factors_all[0]))

def load_config_huice_1030():
    """
    设置10:30模型的回测参数
    :return:
    """
    factors_all1 = [0, 1, 2, 3, 4, 7, 8, 13, 16, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    cn.factors_all = [factors_all1]

    cn.kk_all=[3]                       # 标签频率为9:30,10:00,10:30,11:00,14:30
    cn.num_class = 3
    cn.timestep_all = [3]
    cn.creat_pre = True                 #输出预测明细为True，否则为False
    cn.train_model = True              #是否需要训练

    cn.shujuji='shujuji.csv'            #选择训练和测试数据集
    cn.load_model_dataset = False       #true: 直接从csv文件加载模型的训练集和测试集;  false: 用shujuji生成模型的训练集和测试集

    cn.ww  = True                       #训练集是否需要做筛选
    cn.ww2 = True                      #测试集是否需要做筛选
    cn.start = '2020-01-01'  # 训练集与测试集的总时间段
    cn.end = '2020-12-28'

    cn.ww3=False                        #是否需要筛选掉不需要的时间段
    cn.start2='2018-08-06'              #删除不需要的时间段`
    cn.end2='2016-01-04'

    cn.h2=100                          #训练集的天数
    cn.ratio=2000                       #滚动时测试集的天数
    cn.h3 = cn.ratio                    #测试集的天数
    cn.all_times = cn.h2 + cn.ratio     #每一次训练与预测时，训练集与测试集总的天数

    cn.yinzi2='ROCP_4'              #训练集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2' 训练集的筛选 'OP_CL'
    cn.yinzi3='OP_CL'               #测试集的筛选，包括'MACD','rise_dawn5','ROCP_4','MA_50_ROCP2'  测试集的筛选

    cn.today='2020-12-29-1030'           #合并的数据中滚动训练的文件的保存日期

    cn.trainFileName255 = base_dir +  r'\stock_simulation\Stock_date\stock_result\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                          % (cn.today,cn.yinzi2,cn.shujuji[0:8],cn.ww,cn.ww2,cn.h2,cn.h3,cn.kk_all[0],cn.num_class,len(cn.factors_all[0]))
    cn.trainFileName25 = base_dir +  r'\stock_simulation\Stock_date\stock_result\all_file\%s_%s_%s_ww%s_ww2%s_%s_%s_label%s_calss%s_factor%s' \
                         % (cn.today, cn.yinzi2, cn.shujuji[0:8], cn.ww, cn.ww2, cn.h2, cn.h3, cn.kk_all[0], cn.num_class, len(cn.factors_all[0]))
