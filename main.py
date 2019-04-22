#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: main.py 
@time: 2019-04-22 10:12
@description:
"""
import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

train = pd.read_csv('input/train_data.csv')
test = pd.read_csv('input/test_a.csv')
print(train.shape)
print(test.shape)
df = pd.concat([train, test], keys="ID", axis=0, sort=False)

no_features = ['ID', 'tradeTime', 'tradeMoney', 'city', 'buildYear',
                    'region', 'plate']
categorical_feas = ['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration','communityName']
# 特征工程
# df=pd.get_dummies(df,columns=categorical_feas)
for col in categorical_feas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
features = [fea for fea in train.columns if fea not in no_features + categorical_feas]

train, test = df[:len(train)], df[len(train):]
print(train.shape)
X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values


# 定义评价函数
def evalerror(y_pred, y_true):
    y_true = y_true.get_label()
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'my_score', 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2)


def score(y_true, y_pred):
    return 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2)

N = 5
kf = KFold(n_splits=N, shuffle=True, random_state=2019)
res_list = []
eval_scores = []
print("start：********************************")
start = time.time()

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 参数设置
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eta': 0.01,
              # 'max_depth':4,
              # 'min_child_weight': 6,
              # 'colsample_bytree': 0.7,
              # 'subsample': 0.7,
              # 'eval_metric': 'rmse',
              # 'gamma':0,
              # 'lambda':1,
              # 'alpha ':0，
              'silent': 1
              }
    # 数据
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    evalist = [(dtrain, "train"), (dvalid, "valid")]

    # 模型
    model = xgb.train(params, dtrain,
                      num_boost_round=2000,
                      evals=evalist,
                      early_stopping_rounds=200,
                      feval=evalerror,
                      verbose_eval=50)
    # 模型验证
    pred = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    # 评估
    r2 = score(y_test, pred)
    # r2 = r2_score(y_test, pred)
    print('...........................r2 value:', r2)
    eval_scores.append(r2)
    # 预测
    dtest = xgb.DMatrix(test_data)
    res = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    res_list.append(res)

print('......................validate result mean :', np.mean(eval_scores))
end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 提交结果
mean_r2 = np.mean(eval_scores)
print("mean r2:", mean_r2)
filepath = 'output/xgb_' + str(mean_r2) + '.csv'  # 线下平均分数

# 转为array
res = np.array(res_list)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = pd.DataFrame()
# result['ID'] = test_id
result['p'] = r
result.to_csv(filepath, header=False, index=False, sep=",")
