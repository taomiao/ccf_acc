from sklearn.externals import joblib
import numpy as np
wf = open("TEST_feat.csv","w")
feat = joblib.load("1000lgb_test.feat")
print(len(feat))
for i in range(len(feat)):
    for k in range(20):
        wf.write(str(feat[i][k]+500*k))
        if k == 19:
            wf.write("\n")
        else:
            wf.write(",")

