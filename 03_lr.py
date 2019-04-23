#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 03_lr.py 
@time: 2019-04-22 22:12
@description:
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np
from gen_feas import load_data

train, test, no_features, features = load_data()

X = train[features].values
y = train['tradeMoney'].values
print(y)
test_data = test[features].values

res_list = []
scores_list = []

kf = KFold(n_splits=5, shuffle=True, random_state=2019)


def myscore(y_true, y_pred):
    y_pred=y_pred.get_label()
    return "score",1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2)


for train_index, test_index in kf.split(X, y):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]

    clf = XGBRegressor(booster='gbtree',
                       objective='reg:linear',
                       eval_metric='rmse',
                       n_estimators=3000,
                       n_thead=8)
    clf.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=myscore,
            verbose=True)

    # 验证集测试
    valid_pred = clf.predict(x_valid)
    score = r2_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = clf.predict(test_data)
    res_list.append(pred)
