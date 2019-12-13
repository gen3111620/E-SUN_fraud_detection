import os
import gc
import time
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
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


def threshold_search_fold(y_true, y_proba):

    binary_best_threshold = 0
    binary_best_score = 0
    
    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
 
        binary_score = f1_score(y_true, np.where(y_proba>=threshold , 1 ,0))
        if binary_score > binary_best_score:
            binary_best_threshold = threshold
            binary_best_score = binary_score
            
    recall = recall_score(y_true, np.where(y_proba>=binary_best_threshold , 1 ,0))
    precission = precision_score(y_true, np.where(y_proba>=binary_best_threshold , 1 ,0))
    print('best_threshold_recall:', recall)
    print('best_threshold_precision:', precission)
    
    search_result = {'f1_binary_threshold': binary_best_threshold, 'f1_binary': binary_best_score,}
    return search_result

    
train = pickle.load(open('./train_686.pkl', 'rb')).drop(columns=['bacno_conam_ct', 'hr_highrisk', 'prop_hr_highrisk', 'acqic_target_encoding', 'csmcu_target_encoding', 'mcc_target_encoding'], axis=1)
test = pickle.load(open('./test_686.pkl', 'rb')).drop(columns=['bacno_conam_ct', 'hr_highrisk', 'prop_hr_highrisk', 'acqic_target_encoding', 'csmcu_target_encoding', 'mcc_target_encoding'], axis=1)
sub = pd.read_csv('./submission_test.csv')

# testing code
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
 


SEED = 6089
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'fraud_ind'


params = {
            'n_estimators': 100000,
            'learning_rate': 0.01,
            'boosting_type':'Plain',
            'max_ctr_complexity': 2,
            'eval_metric':'F1',
            'loss_function':'Logloss',
            'random_seed':6089,
            'metric_period':500,
            'od_wait':500,
            # need gpu 
            'task_type':'GPU',
            'depth': 12,
            'gpu_ram_part':0.95,
            'border_count' : 254
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
    print(len(trn_idx),len(val_idx))
    
    estimator = CatBoostClassifier(**params) 
    estimator.fit(
            X.iloc[trn_idx,:],y[trn_idx],
            eval_set=(X.iloc[val_idx,:], y[val_idx]),
#             cat_features=cat_feature,
            use_best_model=True)

    pp_p = estimator.predict_proba(P)
    predictions += pp_p[:,1]/NFOLDS
    
    oof[val_idx] += estimator.predict_proba(X.iloc[val_idx,:])[:,1]
    
    print(threshold_search_fold(y[val_idx], oof[val_idx]))
    gc.collect()

print('OOF AUC:', roc_auc_score(y, oof))


search_resutls = threshold_search(y.values, oof)

sub['fraud_ind'] = np.where(predictions>=search_resutls['f1_micro_threshold'], 1, 0)
# sub['fraud_ind'] = np.where(y_preds>=search_resutls['f1_micro_threshold'], 1, 0)
sub.to_csv("cat.csv", index=False)

print(sub.loc[sub['fraud_ind'] >= search_resutls['f1_micro_threshold']].shape)

pickle.dump(oof, open('cat_6fold_train.pkl', 'wb'))
pickle.dump(predictions, open('cat_6fold_test.pkl', 'wb'))
