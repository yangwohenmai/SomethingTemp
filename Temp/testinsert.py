import SqliteHelper
import numpy as np
import pandas as pd
import datetime as dt

newDF = pd.DataFrame()
if(newDF.empty):
    print("ture")

newlist = [[1,2],[3,4],[5,6]]
print(newlist[0])
for item in newlist:
    print("werwe {0},{1}".format(item[0],item[1]))


newlist = [[1,2],[3,4],[5,6]]
#SqliteHelper.InsertData(newlist)
print(newlist)
newlist = pd.DataFrame(newlist)
print(newlist)
SqliteHelper.InsertDFData(newlist)

newlist = newlist.values.tolist()
print(newlist)
