# -*- coding: utf-8 -*-
import pandas as pd
import re
import nltk
import os
from datetime import datetime

# 遍历文件夹
def walkFile(path,pos=4):#需要手动确定event属于第几位
    #example D:\pythonwork\keyDev\event\2-Announcements\a  
    #2-Announcements index是第四位
    event_path={}
    for root, dirs, files in os.walk(path):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        if len(files)==0:continue #filter path do not containe excel
        #根据路径需要手动修改
        
        #create event filepath dictionary
        event=root.split("\\")[pos]
        if event not in event_path.keys():
            event_path[event]=[]

        # 遍历文件
        for f in files:
            event_path[event].append(os.path.join(root, f))
    return event_path

def getbracketdata(x):
    p1 = re.compile(r'[(]([\d|.|-]*?)[)]')
    r=re.findall(p1, x)
    data=r[0] if len(r)==1 else ""
    return data

def getdata(x):
    #2Hon Hai. (TSEC:2317)  8.56  MSCI Emerging - Market F Price
    x=re.sub(r'\(.*?\)','', x)#最短匹配，可能有多个括号
    p1 = re.compile(r'-{0,1}\d+.{0,1}\d+')
    r=re.findall(p1, x)
    if len(r)>=2:r=[i for i in r if not i.isdigit()] #pick the float value, if have two value
    data=r[0] if len(r)>=1 else ""
    return data

def clean(row):
    p1 = re.compile(r'[(](.*?)[)]')
    r=re.findall(p1, row['Company Name(s)'])
    row['ticker']=r[-1] if len(r)>=1 else "None"
    row['Post Event Return (%)']=getbracketdata(row['Post Event Return (%)'])
    row['Post Event Excess Return vs Benchmark Index']=getdata(row['Post Event Excess Return vs Benchmark Index'])
    row['7 Day Return (%)']=getbracketdata(row['7 Day Return (%)'])
    row['30 Day Return (%)']=getbracketdata(row['30 Day Return (%)'])
    row['90 Day Return (%)']=getbracketdata(row['90 Day Return (%)'])
    row['7 Day Excess Return vs Benchmark Index']=getdata(row['7 Day Excess Return vs Benchmark Index'])
    row['30 Day Excess Return vs Benchmark Index']=getdata(row['30 Day Excess Return vs Benchmark Index'])
    row['90 Day Excess Return vs Benchmark Index']=getdata(row['90 Day Excess Return vs Benchmark Index'])
    row['Pre Event Stock Price ($USD, Historical rate)']=getbracketdata(row['Pre Event Stock Price ($USD, Historical rate)'])
    row['Post Event Stock Price ($USD, Historical rate)']=getbracketdata(row['Post Event Stock Price ($USD, Historical rate)'])
    return row

# 输入目录
event_path = walkFile('D:\pythonwork\keyDev\event')

ret_cols=['Post Event Return (%)',
       'Post Event Excess Return vs Benchmark Index', 
       '7 Day Return (%)',
       '30 Day Return (%)', '90 Day Return (%)',
       '7 Day Excess Return vs Benchmark Index',
       '30 Day Excess Return vs Benchmark Index',
       '90 Day Excess Return vs Benchmark Index',
       'Pre Event Stock Price ($USD, Historical rate)',
       'Post Event Stock Price ($USD, Historical rate)']

#main regression
dt=pd.DataFrame() 
log=[] #store excel size and missing value
for event in event_path.keys():
    for path in event_path[event]:
      cache=pd.read_excel(path,skiprows=7)
      dt = dt.append(cache) # aggregate each event data
    dt=dt.drop(columns=['Key Development Situation'])
    dt=dt.apply(clean,axis=1)  #apply better than iterrows
    dt.to_excel(event+".xlsx",index=False)
    #print clean log
    size=dt.shape[0]
    print(event,size,datetime.now())
    miss_count=[dt[dt[i]==''].shape[0]/size for i in ret_cols]
    log.append([event,size]+miss_count)

#log output
log=pd.DataFrame(log)
log.columns=['event','size']+ret_cols
log.to_excel("eventLog.xlsx",index=False)





#####yours######
res = pd.DataFrame(columns=('guidanceID', 'Key Developments By Date', 'Excel Company ID', 'Key Developments by Type', 'Company Name(s)', 'Key Development Sources'))
for filepath in filelist:
    count  = count + 1
    dt=pd.DataFrame()
    print(filepath)
    cache=pd.read_excel(filepath,header=7)
    dt = dt.append(cache)
    print(dt.head(15))
    dt.insert(0, 'guidanceID', range(1, 1 + len(dt)))
    dt=dt.drop(columns=['Key Development Headline','Key Development Situation'])
    #res = pd.DataFrame(columns=('guidanceID', 'Key Developments By Date', 'Excel Company ID', 'Key Developments by Type', 'Company Name(s)', 'Key Development Sources'))
    #res = pd.DataFrame()
    for index,row in dt.iterrows(): 
        #print(row)
        row = clean(dt.iloc[index])
        #print(row) # 输出每一行
        res = res.append(row)
        #res.to_excel("test.xlsx",index=False)
        #print(res)
        #res.to_excel("result"+ str(count) +".xlsx",index=False)

res.to_excel("result"+ str(count) +".xlsx",index=False)

