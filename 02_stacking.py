#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_stacking.py 
@time: 2019-05-05 20:16
@description:
"""
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import time
from utils import xgb_score, lgb_score
from gen_feas import load_data

# ----------加载数据------------
train, test, no_features, features = load_data()

#
X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values

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
    # 数据结构
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        # 'metric': {'mse','l2', 'l1'},
        # 'metric': {'rmse'},
        'num_leaves': 96,
        'learning_rate': 0.02,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
        'min_data_in_leaf': 18,
        'min_sum_hessian_in_leaf': 0.001,
    }

    print('................Start training {} fold..........................'.format(k + 1))
    # train
    lgb_clf = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=lgb_eval,
                        feval=lgb_score,
                        early_stopping_rounds=100,
                        verbose_eval=100, feature_name=features)

    # 验证集测试
    print('................Start predict .........................')
    valid_pred = lgb_clf.predict(x_valid)
    valid_index.extend(list(test_index))
    valid_list.extend(list(y_valid))
    valid_pred_list.extend(list(valid_pred))
    score = r2_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = lgb_clf.predict(test_data)
    res_list.append(pred)

lgb.plot_importance(lgb_clf, max_num_features=20)
plt.show()

### 特征选择
df = pd.DataFrame(train[features].columns.tolist(), columns=['feature'])
df['importance'] = list(lgb_clf.feature_importance())  # 特征分数
df = df.sort_values(by='importance', ascending=False)
print(list(df['feature'].values))
# 特征排序
df.to_csv("output/feature_score.csv", index=None)  # 保存分数

print('......................validate result mean :', np.mean(scores_list))
end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 线下平均分数
mean_score = np.mean(scores_list)
print("lgb mean score:", mean_score)
filepath = 'output/lgb_' + str(mean_score) + '.csv'  #

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
full_df.to_csv('output/lgb_df.csv', index=None)

# ----------xgb------------

# valid_id = []  # 训练集
# valid_index = []
# valid_list = []
# valid_pred_list = []
#
# res_list = []  # 结果
# scores_list = []
#
# print("start：********************************")
# start = time.time()
# kf = KFold(n_splits=5, shuffle=True, random_state=2019)
# for k, (train_index, test_index) in enumerate(kf.split(X, y)):
#     x_train, y_train = X[train_index], y[train_index]
#     x_valid, y_valid = X[test_index], y[test_index]
#     valid_id.extend(list(train.ID[test_index].values))
#
#     # 参数设置
#     params = {'booster': 'gbtree',
#               'objective': 'reg:linear',
#               'eta': 0.02,
#               # 'max_depth':4,
#               'min_child_weight': 6,
#               'colsample_bytree': 0.7,
#               'subsample': 0.7,
#               # 'eval_metric':'rmse',
#               # 'gamma':0,
#               # 'lambda':1,
#               # 'alpha ':0，
#               'silent': 1
#               }
#     # 数据结构
#     dtrain = xgb.DMatrix(x_train, label=y_train)
#     dvalid = xgb.DMatrix(x_valid, label=y_valid)
#     evallist = [(dtrain, 'train'), (dvalid, 'valid')]  # 'valid-auc' will be used for early stopping
#     # 模型train
#     xgb_clf = xgb.train(params, dtrain,
#                         num_boost_round=1000,
#                         evals=evallist,
#                         early_stopping_rounds=100,
#                         verbose_eval=100)
#
#     # 验证集测试
#     print('................Start predict .........................')
#     valid_pred = xgb_clf.predict(dvalid)
#     valid_index.extend(list(test_index))
#     valid_list.extend(list(y_valid))
#     valid_pred_list.extend(list(valid_pred))
#     score = r2_score(y_valid, valid_pred)
#     print("------------ r2_score:", score)
#     scores_list.append(score)
#
#     # 测试集预测
#     dtest = xgb.DMatrix(test_data)
#     pred = xgb_clf.predict(dtest)
#     res_list.append(pred)
#
# print('......................validate result mean :', np.mean(scores_list))
# end = time.time()
# print("......................run with time: ", (end - start) / 60.0)
# print("over:*********************************")
#
# # 线下平均分数
# mean_score = np.mean(scores_list)
# print("lgb mean score:", mean_score)
# filepath = 'output/xgb_' + str(mean_score) + '.csv'  #
#
# # 提交结果 5折平均
# print("lgb 提交结果...")
# res = np.array(res_list)
# r = res.mean(axis=0)
# result = pd.DataFrame()
# result['p'] = r
# result.to_csv(filepath, header=False, index=False, sep=",")
# # 训练集结果
# print("训练集结果")
# raw_df = pd.read_csv('input/train_data.csv')
# valid_df = pd.DataFrame()
# valid_df['ID'] = valid_id
# valid_df['pred_tradeMoney'] = valid_pred_list
# full_df = pd.merge(raw_df, valid_df, on="ID")
# full_df.to_csv('output/xgb_df.csv', index=None)
#
# # ----------bayes------------
