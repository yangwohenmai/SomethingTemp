import sqlite3
from datetime import datetime
import pandas as pd
 
conn = sqlite3.connect(r'E:\小程序\Python_Test_del\JQStockData.db')
print("Opened database successfully")

def InsertData(data):
    for i in data:
        for j in i:
            conn.execute("INSERT INTO DayData(date,exchange,symbol,open) VALUES (1,2,3,"+str(j)+")")
    conn.commit()
    conn.close()


def InsertListData(data):
    for i in data:
        for j in i:
            conn.execute("INSERT INTO DayData(date,exchange,symbol,open) VALUES (1,2,3,"+str(j)+")")
    conn.commit()
    conn.close()

def InsertDFData(data,exchange,symbol):
    conn = sqlite3.connect(r'E:\小程序\Python_Test_del\JQStockData.db')
    data = data.values.tolist()
    for item in data:
        #print("INSERT INTO MonthData(date,open,close,high,low,volume,money,exchange,symbol) VALUES ({0},{1},{2},{3},{4},{5},{6},{7},{8})".format(item[0].replace("-",""),item[1],item[2],item[3],item[4],item[5],item[6],exchange,symbol))
        conn.execute("INSERT INTO MonthData(date,open,close,high,low,volume,money,exchange,symbol) VALUES ({0},{1},{2},{3},{4},{5},{6},'{7}','{8}')".format(str(item[0]).replace("-",""),item[1],item[2],item[3],item[4],item[5],item[6],exchange,symbol))
    conn.commit()
    conn.close()

def InsertArrayData(data):
    for i in data:
        conn.execute("INSERT INTO DayData(date,exchange,symbol,open) VALUES (1,2,3,"+str(i)+")")
    conn.commit()
    conn.close()