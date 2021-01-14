"""
合并对接淘宝的30min数据，因为QA中30min数据不全
有的股票是2016年开始有30min，有的是2017年开始有30min
"""
import pandas as pd
import os
import config2 as cn

def merge():
    shuju_yuan = r'E:\new file\10_30fenzhong'  # 历史日线数据存放目录
    his_min_data_dir = cn.qa_his_min_data_dir  # 30分钟数据存放目录

    min_files = os.listdir(his_min_data_dir)
    m = 0
    for filename in min_files:
        try:
            m = m + 1

            print("对接中...", m, filename)

            qa_min_file = os.path.join(his_min_data_dir, filename)
            tb_min_file = os.path.join(shuju_yuan, filename)

            qa_min_data = pd.read_table(qa_min_file, sep=',')  # 30分钟数据读取
            tb_min_data = pd.read_table(tb_min_file, sep=',')  # 历史30分钟数据读取

            tb_min_data.rename(columns={'时间': 'datetime', '开盘价': 'open', '最高价': 'high', '最低价': 'low',
                                        '收盘价': 'close', '成交量(手)': 'vol'}, inplace=True)

            qa_min_data = qa_min_data.ix[::, ['datetime', 'open', 'close', 'high', 'low', 'vol']]  # 30分钟数据读取
            tb_min_data = tb_min_data.ix[::, ['datetime', 'open', 'close', 'high', 'low', 'vol']]  # 历史30分钟数据读取

            # 排除重复的数据
            a = qa_min_data['datetime'].tolist()
            b = tb_min_data['datetime'].tolist()
            df6 = set(a) & set(b)
            df7 = set(a) - df6
            data3 = qa_min_data[qa_min_data['datetime'].isin(df7)].sort_index(ascending=True)

            # 合并qa, tb数据
            data4 = pd.concat([tb_min_data, data3], axis=0)
            data4.to_csv(qa_min_file, mode='w', index=False, header=True)
        except:
            print("error: ", filename)

if __name__ == '__main__':
    merge()


