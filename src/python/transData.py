import numpy as np
from sklearn.externals import joblib
import os
import pandas as pd
path = "./TEST_csv"
path_save = "./TEST_npz"
featpath = "./TEST_feat"
if not os.path.exists(path_save):
    os.mkdir(path_save)
    os.mkdir(featpath)
for name in os.listdir(path):
    filename = os.path.join(path,name)
    file = pd.read_csv(filename,header=None).values
    dict = {}
    x = file[:,1:]
    x_app = np.ones([np.shape(x)[0],1])*12460
    x = np.hstack([x,x_app])
    dict["x"] = x
    dict["y"] = np.reshape(file[:,0],[-1,1])
   # print(dict["y"])
    featname = os.path.join(featpath,name)
    feat = pd.read_csv(featname,header=None).values
    dict["feat"] = feat
#    print(feat)
    joblib.dump(dict,os.path.join(path_save,name))