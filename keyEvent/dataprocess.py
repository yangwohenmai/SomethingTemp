# -*- coding: utf-8 -*-
import pandas as pd
import re
import nltk
import os

#load data
files = [f for f in os.listdir('.') if os.path.isfile(f)]
yearfile=["mf/"+i for i in files if "2019" in i]
dt=pd.DataFrame()
for yf in yearfile:
    print(yf)
    cache=pd.read_excel(yf,skiprows=7)
    dt = dt.append(cache)

#attributionDict=[" "+i.strip()+" " for i in open("Attributionlist.txt").readlines()]

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

def labelGuideType(x):
    if "Corporate Guidance - Raised" in x:return "Raised"
    elif "Corporate Guidance - New/Confirmed" in x:return "Confirmed"
    elif "Corporate Guidance - Lowered" in x: return "Lowered"
    elif "Unusual Events" in x:return "Unusual"
    else: return "Others"
    
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
    return row

#def containAttribution(row):
#    attriSen=[]
#    flag='0'
#    text=row['Key Development Situation']
#    sentences=nltk.sent_tokenize(text)
#    for sen in sentences:
#      if any(wrd in sen for wrd in attributionDict):
#         attriSen.append(sen)
#         flag='1'
#    row['attribution']=flag
#    row['attributionSen']="\n".join(attriSen)
#    return row

#dt=dt.apply(clean,axis=1)
#dt=dt.apply(containAttribution,axis=1)
#dt.to_excel("earningGuidance2019.xlsx",index=False)



#load all years
path='C:/Users/hkpu/Dropbox/Raw Data/'
path='C:/Users/admin/Desktop/keyEvent/keyEvent/all event/announcement/2007.1.1-2015.12.31/'
files = [path+f for f in os.listdir(path)]
dt=pd.DataFrame()
for yf in files:
    print(yf)
    cache=pd.read_excel(yf)
    dt = dt.append(cache)
    print(dt.head(5))

dt.insert(0, 'guidanceID', range(1, 1 + len(dt)))
dt=dt.drop(columns=['Key Development Headline','Key Development Situation'])
dt2=dt.sample(100)

dt['Post Event Return (%)'] = pd.to_numeric(dt['Post Event Return (%)'],errors='coerce')
dt['Post Event Excess Return vs Benchmark Index'] = pd.to_numeric(dt['Post Event Excess Return vs Benchmark Index'],errors='coerce')
dt['7 Day Return (%)'] = pd.to_numeric(dt['7 Day Return (%)'],errors='coerce')
dt['30 Day Return (%)'] = pd.to_numeric(dt['30 Day Return (%)'],errors='coerce')
dt['90 Day Return (%)'] = pd.to_numeric(dt['90 Day Return (%)'],errors='coerce')
dt['7 Day Excess Return vs Benchmark Index'] = pd.to_numeric(dt['7 Day Excess Return vs Benchmark Index'],errors='coerce')
dt['30 Day Excess Return vs Benchmark Index'] = pd.to_numeric(dt['30 Day Excess Return vs Benchmark Index'],errors='coerce')
dt['90 Day Excess Return vs Benchmark Index'] = pd.to_numeric(dt['90 Day Excess Return vs Benchmark Index'],errors='coerce')

#import numpy as np
#dt = dt.replace(np.nan, "", regex=True)
#
#dt.to_excel("earningGuidance(2010-2019)214677.xlsx",index=False)
#
##group count
#guidetype=[]
#for idx,row in dt.iterrows():
#    cache=[i.strip() for i in row['Key Developments by Type'].split(";")]
#    guidetype.extend(cache)
#guidetype=pd.DataFrame(guidetype)
#guidetype.columns=['guide']
#guidetype['count']=1
#cache=guidetype.groupby(['guide']).count()
#cache.to_excel("group_count2.xlsx")


import os
dirnum = 0
filenum = 0
path = 'D:\Dropbox\capital IQ all event'
path='C:/Users/admin/Desktop/keyEvent/keyEvent/all event/announcement'

for lists in os.listdir(path):
    sub_path = os.path.join(path, lists)
    print(sub_path)
    if os.path.isfile(sub_path):
        filenum = filenum+1
    elif os.path.isdir(sub_path):
        dirnum = dirnum+1

print('dirnum: ',dirnum)
print('filenum: ',filenum)

import os
path = 'D:\Dropbox\capital IQ all event'
count = 0
for root,dirs,files in os.walk(path):    #遍历统计
      for each in files:
             count += 1   #统计文件夹下文件个数
print(count)               #输出结果



