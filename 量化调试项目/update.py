#!/usr/local/bin/python

# coding :utf-8

from QUANTAXIS.QASU.main import (QA_SU_save_etf_day, QA_SU_save_etf_min,
                                 QA_SU_save_financialfiles,
                                 QA_SU_save_index_day, QA_SU_save_index_min,
                                 QA_SU_save_stock_block, QA_SU_save_stock_day,
                                 QA_SU_save_stock_info,
                                 QA_SU_save_stock_info_tushare,
                                 QA_SU_save_stock_list, QA_SU_save_stock_min,
                                 QA_SU_save_stock_xdxr, QA_SU_save_report_calendar_day,
                                 QA_SU_save_report_calendar_his, QA_SU_save_stock_divyield_day,
                                 QA_SU_save_stock_divyield_his, QA_SU_save_future_min_all)




if __name__ == '__main__':
    QA_SU_save_index_day('tdx')  # 更新指数数据
    QA_SU_save_stock_day('tdx')  # 更新日期数据
    QA_SU_save_stock_xdxr('tdx')
    # QA_SU_save_stock_min('tdx') # 更新各级别分钟数据，这个有点大，硬盘不够100g的需要注释掉
    QA_SU_save_stock_block('tdx')
    QA_SU_save_stock_list('tdx')
    QA_SU_save_stock_info('tdx')