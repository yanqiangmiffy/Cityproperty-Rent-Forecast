#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: ms_interpret.py
@time: 2019-05-21 11:19
@description:
"""
import numpy as np
import time
import pandas as pd
from gen_feas import load_data
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# ----------加载数据------------
train, test, no_features, features = load_data()
#
# #
X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values
seed = 1

# ----------lgb------------
valid_id = []  # 训练集
valid_index = []
valid_list = []
valid_pred_list = []

res_list = []  # 结果
scores_list = []

print("start：********************************")
start = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=2019)
for k, (train_index, test_index) in enumerate(kf.split(X, y)):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]
    valid_id.extend(list(train.ID[test_index].values))
    # model
    print('................Start training {} fold..........................'.format(k + 1))
    ebm = ExplainableBoostingRegressor(n_estimators=50,
                                       random_state=seed)
    ebm.fit(x_train, y_train)  # Works on dataframes and numpy arrays

    # 验证集测试
    print('................Start predict .........................')
    valid_pred = ebm.predict(x_valid)
    valid_index.extend(list(test_index))
    valid_list.extend(list(y_valid))
    valid_pred_list.extend(list(valid_pred))
    score = r2_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = ebm.predict(test_data)
    res_list.append(pred)

print('......................validate result mean :', np.mean(scores_list))
end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 线下平均分数
mean_score = np.mean(scores_list)
print("lgb mean score:", mean_score)
filepath = 'output/ebm_' + str(mean_score) + '.csv'  #

# 提交结果 5折平均
print("lgb 提交结果...")
res = np.array(res_list)
r = res.mean(axis=0)
result = pd.DataFrame()
result['p'] = r
result.to_csv(filepath, header=False, index=False, sep=",")
# 训练集结果
print("训练集结果")
raw_df = pd.read_csv('input/train_data.csv')
valid_df = pd.DataFrame()
valid_df['ID'] = valid_id
valid_df['pred_tradeMoney'] = valid_pred_list
full_df = pd.merge(raw_df, valid_df, on="ID")
full_df.to_csv('output/ebm_df.csv', index=None)
