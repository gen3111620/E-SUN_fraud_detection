import os
import gc
import math
import pickle
import datetime
import warnings
import sys, random
import numpy as np 
import pandas as pd

import seaborn as sns
import lightgbm as lgb
import multiprocessing
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')
tqdm.pandas()

### read csv

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# sub = pd.read_csv('tb-mountain-go/submission_test.csv')

##### 這些特徵 來自於其他py檔案做，因較為耗時，所以另外做 最後merge進來

delta_conam_mean_etymd = pd.read_csv('delta_conam_mean_etymd.csv')
prop_hr_highrisk = pd.read_csv('prop_hr_highrisk.csv')
prop_contp_stocn = pd.read_csv('prop_contp_stocn.csv')
prop_contp_stscd = pd.read_csv('prop_contp_stscd.csv')

ecfg_acc = pd.read_csv('data_ecfg_acc.csv')
ecfg_inf_acc = pd.read_csv('data_ecfg_inc_acc.csv')

fe_day = pd.read_csv('data_ex1d.csv')
fe_hour = pd.read_csv('data_ex1hr.csv')

hr_scity = pd.read_csv('hr_scity_prop.csv')
hr_stocn = pd.read_csv('hr_stocn_prop.csv')


train = pd.merge(left=train, right=delta_conam_mean_etymd, on='txkey')
test = pd.merge(left=test, right=delta_conam_mean_etymd, on='txkey')

train = pd.merge(left=train, right=prop_hr_highrisk, on='txkey')
test = pd.merge(left=test, right=prop_hr_highrisk, on='txkey')

train = pd.merge(left=train, right=prop_contp_stocn, on='txkey')
test = pd.merge(left=test, right=prop_contp_stocn, on='txkey')

train = pd.merge(left=train, right=prop_contp_stscd, on='txkey')
test = pd.merge(left=test, right=prop_contp_stscd, on='txkey')

train = pd.merge(left=train, right=ecfg_acc, on='txkey')
train = pd.merge(left=train, right=ecfg_inf_acc, on='txkey')

test = pd.merge(left=test, right=ecfg_acc, on='txkey')
test = pd.merge(left=test, right=ecfg_inf_acc, on='txkey')

train = pd.merge(left=train, right=fe_day, on='txkey')
train = pd.merge(left=train, right=fe_hour, on='txkey')

test = pd.merge(left=test, right=fe_day, on='txkey')
test = pd.merge(left=test, right=fe_hour, on='txkey')


train = pd.merge(left=train, right=hr_scity, on='txkey')
train = pd.merge(left=train, right=hr_stocn, on='txkey')

test = pd.merge(left=test, right=hr_scity, on='txkey')
test = pd.merge(left=test, right=hr_stocn, on='txkey')

#####

## not good feature, may overfitting 
for col in ['acqic', 'cano', 'contp', 'csmcu', 'etymd',
           'hcefg', 'iterm', 'locdt',
           'mcc', 'mchno', 'scity', 'stocn', 'stscd']:
    new_col_name = col+'_bacno_mean'
    temp_df = pd.concat([train[[col, 'bacno']], test[[col,'bacno']]])
    temp_df.loc[temp_df[col]==-1,col] = np.nan
    temp_df = temp_df.groupby(['bacno'])[col].agg(['mean']).reset_index().rename(columns={'mean': new_col_name})

    temp_df.index = list(temp_df['bacno'])
    temp_df = temp_df[new_col_name].to_dict()  
    
    train[new_col_name] = train['bacno'].map(temp_df).astype('float32')
    test[new_col_name]  = test['bacno'].map(temp_df).astype('float32')
    train[new_col_name].fillna(-1,inplace=True)
    test[new_col_name].fillna(-1,inplace=True)

## nunique feature 
for col in ['acqic', 'mcc', 'mchno', 'scity', 'conam']:
    comb = pd.concat([train[['bacno']+[col]],test[['bacno']+[col]]],axis=0)
    mp = comb.groupby('bacno')[col].agg(['nunique'])['nunique'].to_dict()
    train['bacno_'+col+'_ct'] = train['bacno'].map(mp).astype('float32')
    test['bacno_'+col+'_ct'] = test['bacno'].map(mp).astype('float32')
    print('bacno_'+col+'_ct, ',end='')


## nunique feature 
for col in ['acqic', 'mcc', 'csmcu']:
    comb = pd.concat([train[['cano']+[col]],test[['cano']+[col]]],axis=0)
    mp = comb.groupby('cano')[col].agg(['nunique'])['nunique'].to_dict()
    train['cano_'+col+'_ct'] = train['cano'].map(mp).astype('float32')
    test['cano_'+col+'_ct'] = test['cano'].map(mp).astype('float32')
    print('cano_'+col+'_ct, ',end='')


#####

# rolling feature 

cano_rolling_3day = pd.read_csv('cano_locdt_move_average_3day_noshift.csv')
cano_rolling_3day = cano_rolling_3day.drop(columns=['mean'])
cano_rolling_3day = cano_rolling_3day.fillna(-1)

bacno_rolling_3day = pd.read_csv('bacno_locdt_move_average_3day_noshift.csv')
bacno_rolling_3day = bacno_rolling_3day.drop(columns=['mean'])
bacno_rolling_3day = bacno_rolling_3day.fillna(-1)

train = train.merge(cano_rolling_3day, on=['cano', 'locdt'], how='left')
test = test.merge(cano_rolling_3day, on=['cano', 'locdt'], how='left')

train = train.merge(bacno_rolling_3day, on=['bacno', 'locdt'], how='left')
test = test.merge(bacno_rolling_3day, on=['bacno', 'locdt'], how='left')

#####

# cano & bacno rolling feature
train['cano_rolling_3day_divide_conam'] = train['conam'] / train['cano_locdt_move_average_3day_noshift']
train['bacno_rolling_3day_divide_conam'] = train['conam'] / train['bacno_locdt_move_average_3day_noshift']

test['cano_rolling_3day_divide_conam'] = test['conam'] / test['cano_locdt_move_average_3day_noshift']
test['bacno_rolling_3day_divide_conam'] = test['conam'] / test['bacno_locdt_move_average_3day_noshift']


# stocn fraud ratio, seems target encoding
stocn_fraud_count = train.loc[train['fraud_ind'] == 1].groupby(['stocn', 'fraud_ind'])['txkey'].agg(['count']).reset_index()
stocn_fraud_count['stocn_fraud_ratio_risk'] = stocn_fraud_count['count'] / sum(stocn_fraud_count['count'].values) 
stocn_fraud_count = stocn_fraud_count.drop(columns=['count', 'fraud_ind'])
train = train.merge(stocn_fraud_count, on=['stocn'], how='left')
test = test.merge(stocn_fraud_count, on=['stocn'], how='left')

train = train.fillna(-1)
test = test.fillna(-1)

# mv 7day conam
for col in ['locdt_move_average_7day']:
    for agg_type in ['mean','std']:
        new_col_name = col+'_conam_'+agg_type
        temp_df = pd.concat([train[['locdt', 'conam']], test[['locdt', 'conam']]])
        temp_df = temp_df.groupby(['locdt'])['conam'].agg([agg_type]).shift().rolling(window=7).mean().reset_index().rename(columns={agg_type: new_col_name})
        
        print(temp_df)
        temp_df.index = list(temp_df['locdt'])
        temp_df = temp_df[new_col_name].to_dict()   
    
        train[new_col_name] = train['locdt'].map(temp_df)
        test[new_col_name]  = test['locdt'].map(temp_df)
        


def group_date_weekly(date):
    return math.ceil(date/7)

train['date_weekly'] = train['locdt'].apply(lambda x : group_date_weekly(x))
test['date_weekly'] = test['locdt'].apply(lambda x : group_date_weekly(x))

for df in [train, test]:
    df['weekly_transaction_conam_mean'] = df['conam'] / df.groupby(['date_weekly'])['conam'].transform('mean')
    df['weekly_transaction_conam_std'] = df['conam'] / df.groupby(['date_weekly'])['conam'].transform('std')
    
    df['cano_weekly_transaction_conam_mean'] = df['conam'] / df.groupby(['cano', 'date_weekly'])['conam'].transform('mean')
    df['cano_weekly_transaction_conam_std'] = df['conam'] / df.groupby(['cano', 'date_weekly'])['conam'].transform('std')
    
    df['bacno_weekly_transaction_conam_mean'] = df['conam'] / df.groupby(['bacno','date_weekly'])['conam'].transform('mean')
    df['bacno_weekly_transaction_conam_std'] = df['conam'] / df.groupby(['bacno','date_weekly'])['conam'].transform('std')
    
    df['mcc_weekly_transaction_conam_mean'] = df['conam'] / df.groupby(['mcc', 'date_weekly'])['conam'].transform('mean')
    df['mcc_weekly_transaction_conam_std'] = df['conam'] / df.groupby(['mcc', 'date_weekly'])['conam'].transform('std')
    

## nunique feature 
for col in ['date_weekly']:
    comb = pd.concat([train[['bacno']+[col]],test[['bacno']+[col]]],axis=0)
    mp = comb.groupby('bacno')[col].agg(['nunique'])['nunique'].to_dict()
    train['bacno_'+col+'_ct'] = train['bacno'].map(mp).astype('float32')
    test['bacno_'+col+'_ct'] = test['bacno'].map(mp).astype('float32')
    print('bacno_'+col+'_ct, ',end='')


for col in ['date_weekly']:
    new_col_name = col+'_bacno_mean'
    temp_df = pd.concat([train[[col, 'bacno']], test[[col,'bacno']]])
    temp_df.loc[temp_df[col]==-1,col] = np.nan
    temp_df = temp_df.groupby(['bacno'])[col].agg(['mean']).reset_index().rename(columns={'mean': new_col_name})

    temp_df.index = list(temp_df['bacno'])
    temp_df = temp_df[new_col_name].to_dict()  
    
    train[new_col_name] = train['bacno'].map(temp_df).astype('float32')
    test[new_col_name]  = test['bacno'].map(temp_df).astype('float32')
    train[new_col_name].fillna(-1,inplace=True)
    test[new_col_name].fillna(-1,inplace=True)

train = train.fillna(-999)
test = test.fillna(-999)

train['ovrlt'] = train['ovrlt'].apply(lambda x : 0 if x == 'N' else 1)
test['ovrlt'] = test['ovrlt'].apply(lambda x : 0 if x == 'N' else 1)



# 卡號或歸戶是否曾經盜刷
def fieldFraud(train, test):

    bacno_fraud = set(train[train['fraud_ind']==1]['bacno'])
    cano_fraud = set(train[train['fraud_ind']==1]['cano'])
    
    train['bacno_fieldFraud'] = 0
    train['cano_fieldFraud'] = 0

    banco_list = list()
    cano_list = list()

    ## training data section
    for i in range(1,91):
        print(i)

        train.loc[train['locdt'] == i,'bacno_fieldFraud'] = train[train['locdt'] == i].apply(lambda x : 1 if x['bacno'] in banco_list else 0, axis=1)
        banco_list.extend(list(set(train[(train['locdt'] == i)&(train['fraud_ind']==1)]['bacno'])))

        train.loc[train['locdt'] == i,'cano_fieldFraud'] = train[train['locdt'] == i].apply(lambda x : 1 if x['bacno'] in cano_list else 0, axis=1)
        cano_list.extend(list(set(train[(train['locdt'] == i)&(train['fraud_ind']==1)]['cano'])))

    ## testing data section
    test['bacno_fieldFraud'] = test.apply(lambda x : 1 if x['bacno'] in bacno_fraud else 0, axis=1)
    test['cano_fieldFraud'] = test.apply(lambda x : 1 if x['cano'] in cano_fraud else 0, axis=1)

    return train, test

train, test = fieldFraud(train,test)


for df in [train, test]:
    #當天交易次數
    df['cano_locdt_conam_count'] = df.groupby(['cano', 'locdt'])['conam'].transform('count')


def convert_time_to_string(time):
    s_time = str(int(time))
    l = len(s_time)
    i = 6 - l
    res = '0' * i + s_time
    return res


def change_time_to_sec(date):
    date = convert_time_to_string(date)
    return int(date[:2])*60*60 + int(date[2:4])*60 + int(date[4:])

# 將時間轉為秒
train['sec_time'] = train['loctm'].apply(lambda x : change_time_to_sec(x))
test['sec_time'] = test['loctm'].apply(lambda x : change_time_to_sec(x))


for df in [train, test]:
    # 當天金額重複的筆數 & 比例
    df['cano_locdt_conam_nunique'] = df.groupby(['cano', 'locdt'])['conam'].transform('nunique')

    ind = list(df.loc[df['cano_locdt_conam_count'] <= 1].index.values)
    df.loc[ind, 'cano_locdt_conam_nunique'] = -1

    df['cano_locdt_conam_nunique_percent'] = df['cano_locdt_conam_nunique'] / df['cano_locdt_conam_count']

    df['cano_locdt_conam_nunique_percent_to_mean'] = df['cano_locdt_conam_nunique_percent'] / df.groupby(['cano'])['cano_locdt_conam_nunique_percent'].transform('mean')






def if_first_conam_zero(df):
    if (df['cano'], df['locdt']) in multi_index:
        return 1
    return 0

# train
# 當天第一筆交易金額是否為0

start_time = time()

df_sorted_by_time = train.sort_values(['locdt', 'loctm'], ascending=[True, True])
df_sorted_by_time = df_sorted_by_time.reset_index()

df_cano_day_first = df_sorted_by_time.groupby(['cano', 'locdt']).first()

multi_index = df_cano_day_first.loc[df_cano_day_first.conam == 0].index

train['dayfirst_conam_is_zero'] = train.apply(if_first_conam_zero, axis=1)

    
### 與當天上一筆消費時間間隔
# train['sec_time'] = train['loctm'].apply(lambda x : change_time_to_sec(x))

df_sorted_by_time2 = train.sort_values(['cano', 'locdt', 'loctm'], ascending=[True, True, True])
df_sorted_by_time2 = df_sorted_by_time2.reset_index()

df_sorted_by_time2['last_time_interval'] = df_sorted_by_time2['sec_time'].rolling(window=2).apply(lambda x: x[1] - x[0])

# ind = list(df_sorted_by_time2.loc[df_sorted_by_time2['cano_locdt_conam_count']<=1].index.values)

# df_sorted_by_time2.loc[ind, 'last_time_interval'] = -999

txkey_ind = df_cano_day_first.txkey.values

df_sorted_by_time2.loc[df_sorted_by_time2['txkey'].isin(txkey_ind), 'last_time_interval'] = -1

df_sorted_by_time2 = df_sorted_by_time2.loc[:, ['last_time_interval', 'txkey']]

train = pd.merge(left=train, right=df_sorted_by_time2, on='txkey')

print("--- %s seconds ---" % (time() - start_time))

del df_sorted_by_time, df_cano_day_first, df_sorted_by_time2
gc.collect()


# test
# 當天第一筆交易金額是否為0
start_time = time()

df_sorted_by_time = test.sort_values(['locdt', 'loctm'], ascending=[True, True])
df_sorted_by_time = df_sorted_by_time.reset_index()

df_cano_day_first = df_sorted_by_time.groupby(['cano', 'locdt']).first()

multi_index = df_cano_day_first.loc[df_cano_day_first.conam == 0].index

test['dayfirst_conam_is_zero'] = test.apply(if_first_conam_zero, axis=1)

    
### 與當天上一筆消費時間間隔

df_sorted_by_time2 = test.sort_values(['cano', 'locdt', 'loctm'], ascending=[True, True, True])
df_sorted_by_time2 = df_sorted_by_time2.reset_index()

df_sorted_by_time2['last_time_interval'] = df_sorted_by_time2['sec_time'].rolling(window=2).apply(lambda x: x[1] - x[0])

txkey_ind = df_cano_day_first.txkey.values

df_sorted_by_time2.loc[df_sorted_by_time2['txkey'].isin(txkey_ind), 'last_time_interval'] = -1

df_sorted_by_time2 = df_sorted_by_time2.loc[:, ['last_time_interval', 'txkey']]

test = pd.merge(left=test, right=df_sorted_by_time2, on='txkey')

print("--- %s seconds ---" % (time() - start_time))

del df_sorted_by_time, df_cano_day_first, df_sorted_by_time2
gc.collect()


# for df in [train, test]:
#     # 當天消費時間平均間隔
#     df['todat_first_time_binary_feature'] = df['last_time_interval'].apply(lambda x : 1 if x >= 0 else 0)
    
#     df['last_time_interval_mean'] = df.groupby(['cano', 'locdt'])['last_time_interval'].transform('mean')
    
#     df['last_time_interval_mean_to_mean'] = df['last_time_interval_mean'] / df.groupby(['todat_first_time_binary_feature'])['last_time_interval_mean'].transform('mean')
    
#     df['last_time_interval_to_mean'] = df['last_time_interval'] / df.groupby(['cano', 'todat_first_time_binary_feature'])['last_time_interval'].transform('mean')


## fix todat_last_time_binary_feature and todat_first_time_binary_feature
for df in [train, test]:
    df_sorted_by_time = df.sort_values(['locdt', 'loctm'], ascending=[True, True])
    df_sorted_by_time = df_sorted_by_time.reset_index()

    df_cano_day_last = df_sorted_by_time.groupby(['cano', 'locdt']).tail(1)
    txkey_ind = df_cano_day_last.txkey.values

    df['todat_last_time_binary_feature'] = 0
    df.loc[df['txkey'].isin(txkey_ind), 'todat_last_time_binary_feature'] = 1
    
    # 
    df['todat_first_time_binary_feature'] = df['last_time_interval'].apply(lambda x : 0 if x >= 0 else 1)

    
del df, df_sorted_by_time, df_cano_day_last, txkey_ind
gc.collect()

for df in [train, test]:
    # 當天消費時間平均間隔
    # df['todat_first_time_binary_feature'] = df['last_time_interval'].apply(lambda x : 1 if x >= 0 else 0)
    
    df['last_time_interval_mean'] = df.groupby(['cano', 'locdt'])['last_time_interval'].transform('mean')
    
    df['last_time_interval_mean_to_mean'] = df['last_time_interval_mean'] / df.groupby(['todat_first_time_binary_feature'])['last_time_interval_mean'].transform('mean')
    
    df['last_time_interval_to_mean'] = df['last_time_interval'] / df.groupby(['cano', 'todat_first_time_binary_feature'])['last_time_interval'].transform('mean')


# 11/19
# 跟下一筆時間間隔
# train
df_sorted_by_time = train.sort_values(['locdt', 'loctm'], ascending=[True, True])
df_sorted_by_time = df_sorted_by_time.reset_index()

df_cano_day_last = df_sorted_by_time.groupby(['cano', 'locdt']).tail(1)


# train['sec_time'] = train['loctm'].apply(lambda x : change_time_to_sec(x))

df_sorted_by_time2 = train.sort_values(['cano', 'locdt', 'loctm'], ascending=[True, True, True])
df_sorted_by_time2 = df_sorted_by_time2.reset_index()

df_sorted_by_time2['next_time_interval'] = df_sorted_by_time2['sec_time'].shift(-1).rolling(window=2).apply(lambda x: x[0] - x[1])
df_sorted_by_time2['next_time_interval'] = df_sorted_by_time2['next_time_interval']*-1
txkey_ind = df_cano_day_last.txkey.values

df_sorted_by_time2.loc[df_sorted_by_time2['txkey'].isin(txkey_ind), 'next_time_interval'] = -1

df_sorted_by_time2 = df_sorted_by_time2.loc[:, ['next_time_interval', 'txkey']]

train = pd.merge(left=train, right=df_sorted_by_time2, on='txkey')

# train['next_time_interval'] = train['next_time_interval'].apply(lambda x: (-1)*x if x != -999 else x)
# train.loc[train['txkey'].isin(txkey_ind), 'next_time_interval'] = -1

# test
df_sorted_by_time = test.sort_values(['locdt', 'loctm'], ascending=[True, True])
df_sorted_by_time = df_sorted_by_time.reset_index()

df_cano_day_last = df_sorted_by_time.groupby(['cano', 'locdt']).tail(1)


# train['sec_time'] = train['loctm'].apply(lambda x : change_time_to_sec(x))

df_sorted_by_time2 = test.sort_values(['cano', 'locdt', 'loctm'], ascending=[True, True, True])
df_sorted_by_time2 = df_sorted_by_time2.reset_index()

df_sorted_by_time2['next_time_interval'] = df_sorted_by_time2['sec_time'].shift(-1).rolling(window=2).apply(lambda x: x[0] - x[1])
df_sorted_by_time2['next_time_interval'] = df_sorted_by_time2['next_time_interval']*-1

txkey_ind = df_cano_day_last.txkey.values

df_sorted_by_time2.loc[df_sorted_by_time2['txkey'].isin(txkey_ind), 'next_time_interval'] = -1

df_sorted_by_time2 = df_sorted_by_time2.loc[:, ['next_time_interval', 'txkey']]

test = pd.merge(left=test, right=df_sorted_by_time2, on='txkey')

# test['next_time_interval'] = test['next_time_interval'].apply(lambda x: (-1)*x if x != -999 else x)
# test.loc[test['txkey'].isin(txkey_ind), 'next_time_interval'] = -1

for df in [train, test]:
    df['next_time_interval'] = df['next_time_interval'].apply(lambda x: np.abs(x) if x != -1 else x)

del df, df_sorted_by_time, df_cano_day_last, df_sorted_by_time2, txkey_ind
gc.collect()

# 11/19


for df in [train, test]:
    # 當天消費時間平均間隔
    df['next_time_interval_mean'] = df.groupby(['cano', 'locdt'])['next_time_interval'].transform('mean')
    
    df['next_time_interval_mean_to_mean'] = df['next_time_interval_mean'] / df.groupby(['todat_last_time_binary_feature'])['next_time_interval_mean'].transform('mean')
    # df['last_time_interval_mean_to_std'] = df['last_time_interval_mean'] / df.groupby(['todat_first_time_binary_feature'])['last_time_interval_mean'].transform('std')
    
    df['next_time_interval_to_mean'] = df['next_time_interval'] / df.groupby(['cano', 'todat_last_time_binary_feature'])['next_time_interval'].transform('mean')
    # df['last_time_interval_to_std'] = df['last_time_interval'] / df.groupby(['todat_first_time_binary_feature'])['last_time_interval'].transform('std')
    
    df.loc[pd.isnull(df['next_time_interval_to_mean']), 'next_time_interval_to_mean'] = -999 
    

def modify_last_time_interval(last_time_interval):
    
    if last_time_interval<=30:
        return last_time_interval
    else:
        return 31

for df in [train, test]:

    df['short_last_time_interval'] = df['last_time_interval'].apply(lambda x: modify_last_time_interval(x))
    df['short_next_time_interval'] = df['next_time_interval'].apply(lambda x: modify_last_time_interval(x))



# 3 Fold
# def group_date_month(date):
#     if date <=30:
#         return 1
#     elif date >30 and date <=60:
#         return 2
#     elif date >60 and date <=90:
#         return 3
#     else:
#         return 4

### 6 Fold
def group_date_15_day(date):

    if date <=15:
        return 1
    elif date >15 and date <=30:
        return 2
    elif date >30 and date <=45:
        return 3
    elif date >45 and date <=60:
        return 4
    elif date >60 and date <=75:
        return 5
    elif date >75 and date <=90:
        return 6
    elif date >90 and date <=105:
        return 7
    elif date >105 and date <=120:
        return 8

#### 9Fold
# def group_date_15_day(date):
#     if date <=10:
#         return 1
#     elif date >10 and date <=20:
#         return 2
#     elif date >20 and date <=30:
#         return 3
#     elif date >30 and date <=40:
#         return 4
#     elif date >40 and date <=50:
#         return 5
#     elif date >50 and date <=60:
#         return 6
#     elif date >60 and date <=70:
#         return 7
#     elif date >70 and date <=80:
#         return 8
#     elif date >80 and date <=90:
#         return 9
#     elif date >90 and date <=100:
#         return 10
#     elif date >100 and date <=110:
#         return 11
#     elif date >110 and date <=120:
#         return 12
# train['date_month'] = train['locdt'].apply(lambda x : group_date_month(x))
# test['date_month'] = test['locdt'].apply(lambda x : group_date_month(x))
    
train['date_15_day'] = train['locdt'].apply(lambda x : group_date_15_day(x))
test['date_15_day'] = test['locdt'].apply(lambda x : group_date_15_day(x))


## wrong name meaning ? not fix
for df in [train, test]:
    df['mcc_date_month_transaction_conam_mean'] = df['conam'] / df.groupby(['mcc', 'locdt'])['conam'].transform('mean')



# Frequency encoded for both train and test
for feature in ['etymd', 'csmcu', 'stocn', 'mcc', 'acqic', 'mchno', 'contp', 'acqic', 'stocn', 'scity']:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    

# # max - 該天最大秒數 = 0  min - 該天最小秒數 = 0 取得該天的第一筆交易時間 與 最後一筆交易時間 
for df in [train, test]:
    df['loctm_minus_loctm_max'] = np.abs(df.groupby(['cano', 'locdt'])['sec_time'].transform('max') - df.sec_time)
    df['loctm_minus_loctm_min'] = np.abs(df.groupby(['cano', 'locdt'])['sec_time'].transform('min') - df.sec_time)

# wrong code 
# train['todat_first_time_binary_feature'] = train['loctm_minus_loctm_min'].apply(lambda x : 1 if x == 0 else 0)
# test['todat_first_time_binary_feature'] = test['loctm_minus_loctm_min'].apply(lambda x : 1 if x == 0 else 0)

# train['todat_last_time_binary_feature'] = train['loctm_minus_loctm_max'].apply(lambda x : 1 if x == 0 else 0)
# test['todat_last_time_binary_feature'] = test['loctm_minus_loctm_max'].apply(lambda x : 1 if x == 0 else 0)

for df in [train, test]:
    for agg_type in ['mean', 'std']:
        # group 歸卡帳戶及卡號，計算交易金額平均及標準差，及加入每日時間的交易金額平均及標準差
        df['bacno_cano_group_to_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['bacno', 'cano'])['conam'].transform(agg_type)
        df['bacno_cano_locdt_group_to_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['bacno', 'cano', 'locdt'])['conam'].transform(agg_type)

        df['csmcu_group_to_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['csmcu'])['conam'].transform(agg_type)
        df['csmcu_locdt_group_to_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['csmcu', 'locdt'])['conam'].transform(agg_type)



for df in [train, test]:
    for agg_type in ['mean', 'std']:
        df['first_time_money_{}'.format(agg_type)] = df['conam'] / df.groupby(['cano', 'todat_first_time_binary_feature'])['conam'].transform(agg_type)
        df['last_time_money_{}'.format(agg_type)] = df['conam'] / df.groupby(['cano', 'todat_last_time_binary_feature'])['conam'].transform(agg_type)


for df in [train, test]:
    for agg_type in ['mean', 'std']:
        # 計算交易狀態碼與交易金額的 mean std max min 
        df['stscd_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['stscd'])['conam'].transform(agg_type)
        # 計算消費地國別與交易金額的 mean std max min 
        df['stocn_conam__{}'.format(agg_type)] = df['conam'] / df.groupby(['stocn'])['conam'].transform(agg_type)
        # 計算特店代號與交易金額的 mean std max min 
        df['mchno_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['mchno'])['conam'].transform(agg_type)

        # 計算商店MCC代碼與交易金額的 mean std max min 
        df['mcc_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['mcc'])['conam'].transform(agg_type)


for df in [train, test]:
    
    # 消費地國別＋消費城市＋狀態碼
    df['stocn_scity_stscd'] = df['scity'].astype(str)+'_'+df['stocn'].astype(str)+'_'+df['stscd'].astype(str)
    
    # 消費地國別＋消費地幣別＋消費城市
    df['stocn_csmcu_scity'] = df['stocn'].astype(str)+'_'+df['csmcu'].astype(str)+'_'+df['scity'].astype(str)

    # 歸卡帳戶＋卡號＋交易類別
    df['bacno_cano_contp'] = df['bacno'].astype(str)+'_'+df['cano'].astype(str)+'_'+df['contp'].astype(str)
    
    # 歸卡帳戶＋卡號＋日期＋交易型態
    df['bacno_cano_locdt_etymd'] = df['bacno'].astype(str)+'_'+df['cano'].astype(str)+'_'+df['locdt'].astype(str)+'_'+df['etymd'].astype(str)
    
    # 卡號＋特店代號＋消費地國別
    df['bacno_mchno_stocn'] = df['bacno'].astype(str)+'_'+df['mchno'].astype(str)+'_'+df['stocn'].astype(str)

    # 收單行代碼認證授權 + 特店代號 : reference : https://progressbar.tw/posts/75
    df['acqicn_mchno'] = df['acqic'].astype(str)+'_'+df['mchno'].astype(str)
    
    #  收單行代碼認證授權 + 商店MCC代碼 
    df['acqicn_mcc'] = df['acqic'].astype(str)+'_'+df['mcc'].astype(str)

    # 收單行 ＋ 地區 ＋ MCC : reference https://zi.media/@yidianzixun/post/zyDyH3
    df['acqicn_scity_mcc'] = df['acqic'].astype(str)+'_'+df['scity'].astype(str)+'_'+df['mcc'].astype(str)

    df['contp_stocn_scity'] = df['contp'].astype(str)+'_'+df['stocn'].astype(str)+'_'+df['scity'].astype(str)
    
    df['contp_etymd_stocn'] = df['contp'].astype(str)+'_'+df['etymd'].astype(str)+'_'+df['stocn'].astype(str)
    
    # 10/31 bair
    df['hcefg_acqic_csmcu'] = df['hcefg'].astype(str)+'_'+df['acqic'].astype(str)+'_'+df['csmcu'].astype(str)

    df['hcefg_ecfg_mcc'] = df['hcefg'].astype(str)+'_'+df['ecfg'].astype(str)+'_'+df['mcc'].astype(str)

    df['etymd_mchno_csmcu'] = df['etymd'].astype(str)+'_'+df['mchno'].astype(str)+'_'+df['csmcu'].astype(str)

    df['contp_flg_3dsmk_mcc'] = df['contp'].astype(str)+'_'+df['flg_3dsmk'].astype(str)+'_'+df['mcc'].astype(str)

## label encoding

label_encoder_list = ['contp', 'etymd', 'ecfg', 'iterm', 'flbmk',
                      'flg_3dsmk', 'insfg', 'stscd', 'hcefg','stocn', 'csmcu',
                      'mcc','acqic', 'cano', 'scity',
                      'bacno_mchno_stocn', 'stocn_csmcu_scity',
                      'stocn_scity_stscd', 'acqicn_mchno', 'acqicn_mcc',
                      'bacno_cano_locdt_etymd', 'bacno_cano_contp','acqicn_scity_mcc',
                      'contp_stocn_scity', 'contp_etymd_stocn',
                      'hcefg_acqic_csmcu', 'hcefg_ecfg_mcc', 'etymd_mchno_csmcu', 'contp_flg_3dsmk_mcc'
                     ]
for col in label_encoder_list:
    le = LabelEncoder()
    le.fit(np.concatenate([train[col].values.reshape(-1, 1).astype('str'), test[col].values.reshape(-1, 1).astype('str')]))
    train[col] = le.transform(train[col].values.reshape(-1, 1).astype('str'))
    test[col] = le.transform(test[col].values.reshape(-1, 1).astype('str'))





def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

## finally, we dont use the feature 
train['acqic_target_encoding'], test['acqic_target_encoding'] = target_encode(train["acqic"], 
                                                             test["acqic"], 
                                                             target=train.fraud_ind, 
                                                             min_samples_leaf=100,
                                                             smoothing=10,
                                                             noise_level=0.01)

train['mcc_target_encoding'], test['mcc_target_encoding'] = target_encode(train["mcc"], 
                                                             test["mcc"], 
                                                             target=train.fraud_ind, 
                                                             min_samples_leaf=100,
                                                             smoothing=10,
                                                             noise_level=0.01)
train['csmcu_target_encoding'], test['csmcu_target_encoding'] = target_encode(train["csmcu"], 
                                                             test["csmcu"], 
                                                             target=train.fraud_ind, 
                                                             min_samples_leaf=100,
                                                             smoothing=10,
                                                             noise_level=0.01)


#Frequency encoding
for feature in ['bacno_mchno_stocn', 'stocn_scity_stscd','stocn_csmcu_scity', 'acqicn_mchno','acqicn_mcc',\
                'bacno_cano_locdt_etymd', 'bacno_cano_contp', 'acqicn_scity_mcc', 'contp_stocn_scity', 'contp_etymd_stocn',\
                'hcefg_acqic_csmcu', 'hcefg_ecfg_mcc', 'etymd_mchno_csmcu', 'contp_flg_3dsmk_mcc']:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))



for col in ['bacno_mchno_stocn', 'stocn_csmcu_scity',
            'stocn_scity_stscd', 'acqicn_mchno', 'acqicn_mcc',
            'bacno_cano_locdt_etymd', 'bacno_cano_contp','acqicn_scity_mcc', 'contp_stocn_scity', 'contp_etymd_stocn',
            'hcefg_acqic_csmcu', 'hcefg_ecfg_mcc', 'etymd_mchno_csmcu', 'contp_flg_3dsmk_mcc']:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_conam_'+agg_type
        temp_df = pd.concat([train[[col, 'conam']], test[[col,'conam']]])
        temp_df = temp_df.groupby([col])['conam'].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
        
        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()   
    
        train[new_col_name] = train[col].map(temp_df)
        test[new_col_name]  = test[col].map(temp_df)
        
del temp_df
gc.collect()

### num_same_conam_sameday

num_same_conam_sameday = pd.read_csv('./num_same_conam_sameday.csv')

train = pd.merge(left=train, right=num_same_conam_sameday, on='txkey')
test = pd.merge(left=test, right=num_same_conam_sameday, on='txkey')


for df in [train, test]:
    df['same_conam_sameday_ratio'] = df['num_same_conam_sameday']/df['cano_locdt_conam_count']


## num_bef_zero_conam

num_bef_zero_conam = pd.read_csv('./num_bef_zero_conam.csv').drop(columns=['num_bef_records', 'num_sameday_bef_records'])


train = pd.merge(left=train, right=num_bef_zero_conam, on='txkey')
test = pd.merge(left=test, right=num_bef_zero_conam, on='txkey')


for df in [train, test]:
    df['if_bef_has_zero_conam'] = df['num_bef_has_0_conam'].apply(lambda x : 1 if x >= 1 else 0)
    
train = train.drop(columns=['num_bef_has_0_conam'])
test = test.drop(columns=['num_bef_has_0_conam'])

del num_bef_zero_conam, num_same_conam_sameday
gc.collect()

###

### new test feature 1008
for df in [train, test]:
    for agg_type in ['mean', 'std']:
        # 歸卡 卡號 交易型態 網路交易註記 狀態碼
        df['bacno_cano_etymd_ecfg_stscd_to_conam_{}'.format(agg_type)] = df['conam'] / df.groupby(['bacno', 'cano', 'etymd', 'ecfg', 'stscd'])['conam'].transform(agg_type)


train['cents'] = (train['conam'] - np.floor(train['conam'])).astype('float32')
test['cents'] = (test['conam'] - np.floor(test['conam'])).astype('float32')

for df in [train,test]:
    df['conam_acqic_mean'] = df.groupby(['acqic'])['conam'].transform('mean')
    df['conam_acqic_std'] = df.groupby(['acqic'])['conam'].transform('std')
    
    df['conam_cano_acqic_mean'] = df.groupby(['cano', 'acqic'])['conam'].transform('mean')
    df['conam_cano_acqic_std'] = df.groupby(['cano', 'acqic'])['conam'].transform('std')


from sklearn.preprocessing import LabelEncoder
# 11/12
for df in [train, test]:
    df['ecfg_mcc'] = df['ecfg'].astype(str)+'_'+df['mcc'].astype(str)
    
    df['stscd_ecfg_acqic'] = df['stscd'].astype(str)+'_'+df['ecfg'].astype(str)+'_'+df['acqic'].astype(str)
    
    df['ecfg_stocn_scity'] = df['ecfg'].astype(str)+'_'+df['stocn'].astype(str)+'_'+df['scity'].astype(str)
    
label_encoder_list = [
                      'ecfg_mcc', 'stscd_ecfg_acqic', 'ecfg_stocn_scity'
                     ]
for col in label_encoder_list:
    le = LabelEncoder()
    le.fit(np.concatenate([train[col].values.reshape(-1, 1).astype('str'), test[col].values.reshape(-1, 1).astype('str')]))
    train[col] = le.transform(train[col].values.reshape(-1, 1).astype('str'))
    test[col] = le.transform(test[col].values.reshape(-1, 1).astype('str'))
    
#將上面字串串接的feature count values 
for feature in [
                'ecfg_mcc', 'stscd_ecfg_acqic', 'ecfg_stocn_scity'
               ]:
    # Count encoded for both train and test
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

for col in [
            'ecfg_mcc', 'stscd_ecfg_acqic', 'ecfg_stocn_scity'
           ]:
    for agg_type in ['mean']:
        new_col_name = col+'_conam_'+agg_type
        temp_df = pd.concat([train[[col, 'conam']], test[[col,'conam']]])
        temp_df = temp_df.groupby([col])['conam'].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
        
        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()   
    
        train[new_col_name] = train[col].map(temp_df)
        test[new_col_name]  = test[col].map(temp_df)



train = train.replace([np.inf], 999)
test = test.replace([np.inf], 999)

train = train.replace([-np.inf], -999)
test = test.replace([-np.inf], -999)

train = train.fillna(-999)
test = test.fillna(-999)
gc.collect()


# https://www.kaggle.com/alexeykupershtokh/safe-memory-reduction/notebook
def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=-999):
    """
    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
    avg_loss_limit - same but calculates avg throughout the series.
    na_loss_limit - not really useful.
    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
    """
    is_float = str(col.dtypes)[:5] == 'float'
    na_count = col.isna().sum()
    n_uniq = col.nunique(dropna=False)
    try_types = ['float16', 'float32']

    if na_count <= na_loss_limit:
        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

    for type in try_types:
        col_tmp = col

        # float to int conversion => try to round to minimize casting error
        if is_float and (str(type)[:3] == 'int'):
            col_tmp = col_tmp.copy().fillna(fillna).round()

        col_tmp = col_tmp.astype(type)
        max_loss = (col_tmp - col).abs().max()
        avg_loss = (col_tmp - col).abs().mean()
        na_loss = np.abs(na_count - col_tmp.isna().sum())
        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
            return col_tmp

    # field can't be converted
    return col

def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)
        
        # numerics
        if col_type in numerics:
            df[col] = sd(df[col])

        # strings
        if (col_type == 'object') and obj_to_cat:
            df[col] = df[col].astype('category')
        
        if verbose:
            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')
        new_na_count = df[col].isna().sum()
        if (na_count != new_na_count):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')
        new_n_uniq = df[col].nunique(dropna=False)
        if (n_uniq != new_n_uniq):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    if verbose:
        print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df



train = reduce_mem_usage_sd(train, verbose=True)
test = reduce_mem_usage_sd(test, verbose=True)


pickle.dump(train, open('train_686.pkl', 'wb'))
pickle.dump(test, open('test_686.pkl', 'wb'))

print('train features {}'.format(len(list(train.columns))))
print('test features {}'.format(len(list(test.columns))))

