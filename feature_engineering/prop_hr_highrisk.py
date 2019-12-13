import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from sklearn import preprocessing
import xgboost as xgb
from xgboost import plot_importance

from sklearn.model_selection import KFold

# from sklearn.model_selection import train_test_split
# import lightgbm as lgb

import os

train = pd.read_csv("./train.csv") # (1521787, 23)
test = pd.read_csv("./test.csv") # (421665, 22)
# sub = pd.read_csv("/submission_test.csv") # (421665, 2)

X_train = train.drop(["fraud_ind"], axis=1) # (1521787, 21)
y_train = train[["txkey","fraud_ind"]].copy() # (1521787,)
X_test = test.copy() # (421665, 21)

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# Label Encoding
object_count = 0
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        object_count += 1
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

print("the numbers of the object: ", object_count)

data = pd.concat([X_train, X_test]) # (1943452, 22)

def to_hr(t):
    time = str(int(t))
    length = len(time)
    time_str_6 = str(0)*(6-length) + time
    hour = time_str_6[:2]
    return hour

data["hr"] = data["loctm"].apply(to_hr).astype("int")

data["hr_highrisk"] = data["hr"].apply(lambda x: 1 if x < 7 or x > 23 else 0)

df1 = data.groupby(["cano", "stocn"]).agg({"hr_highrisk":"sum"}).reset_index()
df2 = data.groupby(["cano"]).size().reset_index(name="total_count")

df3 = df1.merge(df2)

df3["prop_hr_highrisk"] = df3["hr_highrisk"]/df3["total_count"]
df4 = data.merge(df3[["cano", "stocn", "prop_hr_highrisk"]], on=["cano", "stocn"], how="outer")

df4[["txkey", "hr_highrisk", "prop_hr_highrisk"]].to_csv("prop_hr_highrisk.csv", index=False)