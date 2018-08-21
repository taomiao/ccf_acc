import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sparse
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from sklearn.externals import joblib

rejlist = (0, 1)
if os.path.exists("diclist.dict"):
    diclist = joblib.load("diclist.dict")
else:
    File = open("train.set.txt")
    diclist = {}

    for i in range(41):
        diclist[i] = {}
    for line in File:
        for i,item in enumerate(line.split(",")):
            if i in rejlist:
                continue
            diclist[i][item] = diclist[i].setdefault(item,0) + 1
    joblib.dump(diclist,"diclist.dict")
cnt = 0
dict_5000 = {}
if os.path.exists("dict_5000.dict"):
    dict_5000 = joblib.load("dict_5000.dict")
else:
    for i in diclist:
        if i in rejlist:
            continue
        dict_5000[str(i)+"_other"] = cnt
        cnt = cnt + 1
        for item in diclist[i]:
            if diclist[i][item] > 5000:
                dict_5000[str(i) + "_"+item] = cnt
                cnt = cnt + 1
    joblib.dump(dict_5000,"dict_5000.dict")
ReadFile = open("test.set.a.no.label.txt")
WriteFile = open("test_v2.csv","w")
for line in ReadFile:
    saver = ""
    for i, item in enumerate(line.split(",")):
        if i in rejlist:
	    if i == 1:
		saver = item
            continue
        key = str(i)+"_"+str(item)
        if not key in dict_5000:
            key = str(i)+"_other"
        val = dict_5000[key]
        saver+=","+str(val)
    saver+="\n"
    WriteFile.write(saver)
ReadFile.close()
WriteFile.close()
