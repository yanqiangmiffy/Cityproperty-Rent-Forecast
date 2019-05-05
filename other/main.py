# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")
import copy
import os
import gc
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import xgb_score
from gen_feas import load_data

train, test, no_features, features = load_data()

label = train['tradeMoney']
train = train[features]
test = test[features]


def get_result(train, test, label, my_model, need_sca=False, splits_nums=5):
    oof = np.zeros(train.shape[0])
    sub = pd.DataFrame()
    train = train.values
    test = test.values
    score_list = []
    label = np.array(label)
    test = xgb.DMatrix(test)
    k_fold = KFold(n_splits=splits_nums, shuffle=True, random_state=2019)
    for index, (train_index, test_index) in enumerate(k_fold.split(train)):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = my_model(X_train, y_train, X_test, y_test)

        X_test = xgb.DMatrix(X_test)
        vali_pre = model.predict(X_test)
        oof[test_index] = vali_pre
        score = r2_score(y_test, vali_pre)
        score_list.append(score)

        print(y_test)
        print(vali_pre)

        pred_result = model.predict(test)
        sub['tradeMoney'] = pred_result

        if index == 0:
            re_sub = copy.deepcopy(sub)
        else:
            re_sub['tradeMoney'] = re_sub['tradeMoney'] + sub['tradeMoney']

    re_sub['tradeMoney'] = re_sub['tradeMoney'] / splits_nums

    print('score list:', score_list)
    print(np.mean(score_list))
    return re_sub, oof


def xgb_model(X_train, y_train, X_test, y_test):
    xgb_params = {'eta': 0.01,
                  'max_depth': 6,
                  'subsample': 0.5,
                  'colsample_bytree': 0.5,
                  'alpha': 0.2,
                  'objective': 'reg:gamma',
                  'eval_metric': 'mae',
                  'silent': 1,
                  'nthread': -1
                  }

    trn_data = xgb.DMatrix(X_train, y_train)
    val_data = xgb.DMatrix(X_test, y_test)
    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]

    clf = xgb.train(dtrain=trn_data, num_boost_round=100000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, feval=xgb_score, params=xgb_params)

    return clf


def xgb_model_re(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(colsample_bytree=0.3,
                             # objective= reg:gamma,
                             eval_metric='mae',
                             gamma=0.0,
                             learning_rate=0.01,
                             max_depth=4,
                             min_child_weight=1.5,
                             n_estimators=1668,
                             reg_alpha=1,
                             reg_lambda=0.6,
                             subsample=0.2,
                             seed=42,
                             silent=1)

    model = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=xgb_score,
                      early_stopping_rounds=30,
                      verbose=1)
    return model


def xgb_model_bs(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(max_depth=8,
                             learning_rate=0.1,
                             objective="reg:linear",
                             eval_metric='rmse',
                             n_estimators=3115,
                             colsample_bytree=0.6,
                             reg_alpha=3,
                             reg_lambda=2,
                             gamma=0.6,
                             subsample=0.7,
                             silent=1,
                             n_jobs=-1)
    model = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=xgb_score,
                      early_stopping_rounds=100,
                      verbose=1)

    return model


sub, oof = get_result(train, test, label, xgb_model, need_sca=True, splits_nums=3)
sub[['tradeMoney']].to_csv('output/xgb.csv', index=None, header=False)
