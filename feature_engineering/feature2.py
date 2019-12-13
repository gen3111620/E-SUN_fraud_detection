import numpy as np 
import pandas as pd 
from sklearn import preprocessing


# delta_conam_mean_etymd
# prop_hr_highrisk
# prop_contp_stocn
# prop_contp_stscd
# ecfg_acc
# ecfg_inf_acc
# fe_day
# fe_hour
# hr_scity
# hr_stocn

train = pd.read_csv("train.csv") # (1521787, 23)
test = pd.read_csv("test.csv") # (421665, 22)
# sub = pd.read_csv("submission_test.csv") # (421665, 2)

X_train = train.drop(["fraud_ind"], axis=1) # (1521787, 21)
y_train = train[["txkey","fraud_ind"]].copy() # (1521787,)
X_test = test.copy() # (421665, 21)

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

data = pd.concat([X_train, X_test]) # (1943452, 22)


# prop_contp_stscd
df1 = data.groupby(["cano","contp", "stscd"]).size().reset_index(name="counts")
df2 = data.groupby(["cano","contp"]).size().reset_index(name="all_counts")
df3 = df1.merge(df2)
df3["prop_contp_stscd"] = df3["counts"]/df3["all_counts"]
df4 = data.merge(df3, how="outer")
prop_contp_stscd = df4[["txkey", "prop_contp_stscd"]]
prop_contp_stscd.to_csv('./prop_contp_stscd.csv', index=False)

# prop_contp_stocn
df1 = data.groupby(["cano","contp", "stocn"]).size().reset_index(name="counts")
df2 = data.groupby(["cano","contp"]).size().reset_index(name="all_counts")
df3 = df1.merge(df2)
df3["prop_contp_stocn"] = 1/(df3["counts"]/df3["all_counts"])
df4 = data.merge(df3, how="outer")
prop_contp_stocn = df4[["txkey", "prop_contp_stocn"]]
prop_contp_stocn.to_csv('./prop_contp_stocn.csv', index=False)

# delta_conam_mean_etymd
df1 = train.groupby(["cano", "etymd"]).agg({"conam":"mean"}).reset_index()
df1.columns = ["cano", "etymd", "conam_mean"]
df3 = data.merge(df1, how="outer").fillna(0)
df3["delta_conam_mean_etymd"] = df3["conam"] - df3["conam_mean"]
delta_conam_mean_etymd = df3[["txkey", "delta_conam_mean_etymd"]]
delta_conam_mean_etymd.to_csv('./delta_conam_mean_etymd.csv', index=False)

