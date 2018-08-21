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
def f(x,counter_dict,threshold):
    if counter_dict[x] < threshold:
        return "0"
    return x

if os.path.exists("train_1000.npz") and os.path.exists("test_1000.npz"):
    train = sparse.load_npz("train_1000.npz")
    test = sparse.load_npz("test_1000.npz")
    tag = pd.read_csv("test.set.a.no.label.txt",header = None)[0]
    saver = pd.DataFrame()
    saver["id"] = tag
    print("Load Finished")
else:
    test_raw = pd.read_csv("test.set.a.no.label.txt",header = None)
    print("test finish")
    train_raw = pd.read_csv("train.set.txt",header = None)
    print("train finish")
    saver = pd.DataFrame()
    saver["id"] = test_raw[0]
    split_point = len(test_raw)
    data = pd.concat([test_raw,train_raw])
    del train_raw
    del test_raw
    gc.collect()

    data[1] = data[1].fillna(-1)
    data = data.drop(0,axis=1)
    data.columns = list(range(40))

    for i in range(1,14):
        data[i] = data[i].fillna(np.min(data[i])-1)
    print("step 0 finish")

    for i in range(14,40):
        data[i] = data[i].fillna("-1")
        cd = dict(Counter(data[i]))
        data[i] = data[i].fillna("0").apply(lambda x:f(x,cd,1000))
        del cd
        gc.collect()

    print("step 1 finish")
    onehotfeature = []
    for i in range(14,40):
        enc = LabelEncoder()
        data[i] = enc.fit_transform(data[i])
        enc = OneHotEncoder()
        enc.fit(data[i].values.reshape(-1,1))
        onehotfeature.append(enc.transform(data[i].values.reshape(-1,1)))
        del enc
        gc.collect()
        #onehotfeature += enc.fit_transform(data[i])

    print("step 2 finish")
    onehot = sparse.hstack(onehotfeature)
    dense = data.values[:,:14]
    alldata = sparse.hstack([dense,onehot])
    alldata = sparse.csr_matrix(alldata)
    train = alldata[split_point:]
    test = alldata[:split_point,1:]
    sparse.save_npz("train_1000.npz",train)
    sparse.save_npz("test_1000.npz",test)

clf = joblib.load("1000.lgb")
print(clf.predict(test))
trainfeat = clf.predict_proba(train[:,1:],pred_leaf=True)
joblib.dump(trainfeat,"1000lgb_train.feat")
testfeat = clf.predict_proba(test,pred_leaf=True)
joblib.dump(testfeat,"1000lgb_test.feat")