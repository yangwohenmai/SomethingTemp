#递归查询文件夹下所有文件数量
# -*- coding: utf-8 -*-
import os
path="D:\pythonwork\dataVis" 
num_dirs = 0 #路径下文件夹数量
num_files_rec = 0 #路径下文件数量,包括子文件夹里的文件数量
#root list of all folder path under root path
#dirs list only folder name, could recursion 一级文件夹 ['file1', 'file2'], file1/file2-1/ 二级文件夹 ['file2-1']
#files list all files(without folder) in each folder
excel_count=0
for root,dirs,files in os.walk(path):    #遍历统计
        #count特定文件列表数量
        excel=[i for i in files if i.endswith(".xls")]
        excel_count+=len(excel)
        #count问件数量
        num_files_rec+=len(files)
        #count 文件夹的数量
        for name in dirs:
                num_dirs += 1
                # print(os.path.join(root,name))
print(excel_count) #excel file count
print(num_dirs) #文件夹数量
print(num_files_rec) #路径下文件数量,包括子文件夹里的文件数量



#other potential job
fund holding portfolio analysis-following smart money
corporate disclosure
insider trading
financial news and market return
supply chain data

