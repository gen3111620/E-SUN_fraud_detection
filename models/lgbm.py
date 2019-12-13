import os
import gc
import time
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GroupKFold

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def threshold_search(y_true, y_proba):

    micro_best_threshold = 0
    micro_best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
 
        micro_score = f1_score(y_true, np.where(y_proba>=threshold , 1 ,0))
        if micro_score > micro_best_score:
            micro_best_threshold = threshold
            micro_best_score = micro_score
    search_result = {'f1_micro_threshold': micro_best_threshold, 'f1_micro': micro_best_score}
    return search_result


    
train = pickle.load(open('./train_686.pkl', 'rb')).drop(columns=['bacno_conam_ct', 'hr_highrisk', 'prop_hr_highrisk', 'acqic_target_encoding', 'csmcu_target_encoding', 'mcc_target_encoding'], axis=1)
test = pickle.load(open('./test_686.pkl', 'rb')).drop(columns=['bacno_conam_ct', 'hr_highrisk', 'prop_hr_highrisk', 'acqic_target_encoding', 'csmcu_target_encoding', 'mcc_target_encoding'], axis=1)
sub = pd.read_csv('./submission_test.csv')

# train = train.sample(n=20000).reset_index(drop=True)

train = train.fillna(-999)
test = test.fillna(-999)

features_columns = list(train.columns)
not_use_feature_columns = [ 'fraud_ind', 'txkey','sec_time',
                            'todat_first_time_binary_feature', 'todat_last_time_binary_feature',
                           'loctm_minus_loctm_max', 'loctm_minus_loctm_min',
                            'cano_locdt_conam_nunique_percent', 'cano_locdt_conam_nunique',
                           'last_time_interval_mean', 'last_time_interval',
                           'next_time_interval_mean', 'next_time_interval',
                           'bacno_locdt_move_average_3day_noshift', 'cano_locdt_move_average_3day_noshift',
                           'bacno', 'cano',
                         ]

for i in not_use_feature_columns:
    print(i)
    features_columns.remove(i)
 


SEED = 777
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'fraud_ind'


params = {
            'objective':'binary',
            'boosting_type':'gbdt',
            'metric':'auc',
            'n_jobs':-1,
            'learning_rate':0.01,
            'num_leaves': 2**8,
            'max_depth':-1,
            'tree_learner':'serial',
            'colsample_bytree': 0.4,
            'subsample_freq':1,
            'subsample':0.8,
            'n_estimators': 2000,
            'max_bin':255,
            'verbose':-1,
            'seed': SEED,
            'early_stopping_rounds':200, 
        } 

print(f'The total of features : {len(features_columns)}')

NFOLDS = 6
feature_importances = pd.DataFrame()
feature_importances['feature'] = train[features_columns].columns

folds = GroupKFold(n_splits=NFOLDS)

X,y = train[features_columns], train[TARGET]    
P = test[features_columns] 
split_groups = train['date_15_day']

predictions = np.zeros(len(test))
oof = np.zeros(len(train))
del train, test
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
    print('Fold:',fold_)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
    vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]

    print(len(tr_x),len(vl_x))
    tr_data = lgb.Dataset(tr_x, label=tr_y)
    vl_data = lgb.Dataset(vl_x, label=vl_y)  

    estimator = lgb.train(
        params,
        tr_data,
        valid_sets = [tr_data, vl_data],
        verbose_eval = 200,
#         categorical_feature = categorical_feature
    )   

    predictions += estimator.predict(P) /NFOLDS
    

    oof[val_idx] += estimator.predict(vl_x)

    feature_importances['fold_{}'.format(fold_ + 1)] = estimator.feature_importance()
    del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
    gc.collect()

print('OOF AUC:', roc_auc_score(y, oof))



# plot feature importances
# feature_importances.to_csv('feature_importances.csv')
# feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(NFOLDS)]].mean(axis=1)
# # feature_importances.to_csv('feature_importances.csv')

# plt.figure(figsize=(16, 16))
# sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(100), x='average', y='feature');
# plt.title('100 TOP feature importance over folds average');

search_resutls = threshold_search(y.values, oof)

sub['fraud_ind'] = np.where(predictions>=search_resutls['f1_micro_threshold'], 1, 0)
# sub['fraud_ind'] = np.where(y_preds>=search_resutls['f1_micro_threshold'], 1, 0)
sub.to_csv("lgbm.csv", index=False)

print(sub.loc[sub['fraud_ind'] >= search_resutls['f1_micro_threshold']].shape)

pickle.dump(oof, open('lgbm_6fold_train.pkl', 'wb'))
pickle.dump(predictions, open('lgbm_6fold_test.pkl', 'wb'))
