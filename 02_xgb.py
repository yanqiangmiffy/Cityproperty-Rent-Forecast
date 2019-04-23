#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_xgb.py 
@time: 2019-04-23 23:23
@description:
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from gen_feas import load_data
from utils import my_score
import time

train, test, no_features, features = load_data()

X = train[features].values
y = train['tradeMoney'].values
print(y)
test_data = test[features].values

res_list = []
scores_list = []

kf = KFold(n_splits=5, shuffle=True, random_state=2019)


def myscore(y_true, y_pred):
    y_pred = y_pred.get_label()
    return "score", 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2)


print("start：********************************")
start = time.time()
for train_index, test_index in kf.split(X, y):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]

    clf = XGBRegressor(n_estimators=7000,
                       max_depth = 15,
                       reg_lambda = 0.05,
                       reg_alpha = 0.01,
                       colsample_bytree=0.85,
                       min_child_samples=24,
                       num_leaves = 70)
    clf.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=myscore,
            early_stopping_rounds=100,
            verbose=True)

    # 验证集测试
    valid_pred = clf.predict(x_valid)
    score = r2_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = clf.predict(test_data)
    res_list.append(pred)

print('......................validate result mean :', np.mean(scores_list))
end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 11.5折结果均值融合，并保存文件
mean_auc = np.mean(scores_list)
print("mean auc:", mean_auc)
filepath = 'output/xgb_' + str(mean_auc) + '.csv'  # 线下平均分数

# 转为array
res = np.array(res_list)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = pd.DataFrame()
# result['ID'] = test_id
result['p'] = r * 100
result.to_csv(filepath, header=False, index=False, sep=",")
