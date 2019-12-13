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
sub = pd.read_csv("./submission_test.csv") # (421665, 2)

X_train = train.drop(["fraud_ind"], axis=1) # (1521787, 21)
y_train = train[["txkey","fraud_ind"]].copy() # (1521787,)
X_test = test.copy() # (421665, 21)

del train, test

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

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

def to_date(t):
    if t <= 31:
        month = 3
        day = t
    elif t <= 61:
        month = 4
        day = t-31
    elif t <= 92:
        month = 5
        day = t-61
    else:
        month = 6
        day = t-92
    return "2018/" + str(month) + "/" + str(day)

def to_time(t):
    time = str(int(t))
    length = len(time)
    time_str_6 = str(0)*(6-length) + time
    hour = time_str_6[:2]
    minute = time_str_6[2:4]
    second = time_str_6[4:]
    return "-" + hour + ":" + minute + ":" + second


data["date"] = data["locdt"].apply(to_date)
data["time"] = data["loctm"].apply(to_time)
data["datetime"] = data["date"] + data["time"]

data["datetime"] = pd.to_datetime(data["datetime"], format="%Y/%m/%d-%H:%M:%S")

data.drop(["date", "time"], axis=1 , inplace=True)

df_list = list()

for person in list(set(data.bacno)):
    df_person = data[data.bacno == person]
    for card in list(set(df_person.cano)):
        df_card = df_person[df_person.cano == card]
        def accumulate(t):
            global df_card
            datetime_to = t
            datetime_from = t - pd.offsets.DateOffset(days=121)
#             btw = df_card["datetime"].between(datetime_from , datetime_to)
            btw_1s = df_card["datetime"].between(datetime_from , datetime_to-pd.offsets.DateOffset(seconds=1))
            acc_ecfg = df_card[btw_1s].ecfg.sum()/df_card[btw_1s].ecfg.count()
            return acc_ecfg
        df_card["ecfg_acc"] = df_card["datetime"].apply(accumulate)
#         df_card[["counts_ex1hr", "cost_mean_ex1hr", "cost_med_ex1hr"]] = pd.DataFrame(df_card["datetime"].apply(ex1hr).values.tolist(), index=df_card.index)
        # save
        df_list.append(df_card)

data = pd.concat(df_list)
data.fillna(0, inplace=True)

data = data[["txkey", "ecfg_acc"]]

print(data.shape)
data.to_csv("data_ecfg_acc.csv", index=False)