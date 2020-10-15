# -*- coding: utf-8 -*-
import pandas as pd
import re
import nltk
import os


# 遍历文件夹
def walkFile(file):
    list1 =[]
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            #print(os.path.join(root, f))
            list1.append(os.path.join(root, f))

        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))
            #list1.append(os.path.join(root, d))
    print(list1)
    return list1

def labelGuideType(x):
    if "Corporate Guidance - Raised" in x:return "Raised"
    elif "Corporate Guidance - New/Confirmed" in x:return "Confirmed"
    elif "Corporate Guidance - Lowered" in x: return "Lowered"
    elif "Unusual Events" in x:return "Unusual"
    else: return "Others"

def getbracketdata(x):
    p1 = re.compile(r'[(]([\d|.|-]*?)[)]')
    r=re.findall(p1, x)
    data=r[0] if len(r)==1 else "None"
    return data

def getdata(x):
    #2Hon Hai. (TSEC:2317)  8.56  MSCI Emerging - Market F Price
    x=re.sub(r'\(.*?\)','', x)#最短匹配，可能有多个括号
    p1 = re.compile(r'-{0,1}\d+.{0,1}\d+')
    r=re.findall(p1, x)
    if len(r)>=2:r=[i for i in r if not i.isdigit()] #pick the float value, if have two value
    data=r[0] if len(r)>=1 else "None"
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
    row['GuideType']=labelGuideType(row['Key Developments by Type'])
    #print(row)
    return row



# 输入目录
filelist = walkFile('E:/MyGit/SomethingTemp/keyEvent/all event/announcement/')
count = 0
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


