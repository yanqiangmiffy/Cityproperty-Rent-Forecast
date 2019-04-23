#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_xgb.py
@time: 2019-04-21 23:36
@description:
"""
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


train = pd.read_csv('input/train_data.csv')
test = pd.read_csv('input/test_a.csv')
print(len(train), len(test))
df = pd.concat([train, test], keys="ID", axis=0, sort=True)
import datetime
now = datetime.datetime.now()
df['tradeTime']=pd.to_datetime(df['tradeTime'])
df['interval']=(now-df['tradeTime']).dt.days
no_features = ['ID', 'tradeTime', 'tradeMoney',]

categorical_feas = ['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration', 'city','buildYear', 'region', 'plate', 'communityName']
# df=pd.get_dummies(df,columns=categorical_feas)
for col in categorical_feas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

sd_cols=['totalTradeMoney','totalTradeArea','tradeMeanPrice','totalNewTradeMoney','tradeNewMeanPrice','supplyLandArea','tradeLandArea','landTotalPrice','landMeanPrice','totalWorkers','residentPopulation']
sd=MinMaxScaler()
df[sd_cols]=sd.fit_transform(df[sd_cols])

train, test = df[:len(train)], df[len(train):]
# train=pd.get_dummies(train,columns=categorical_feas)
# test=pd.get_dummies(test,columns=categorical_feas)


features = [fea for fea in train.columns if fea not in no_features]
print(features)
# train.head().to_csv('demo.csv')

# 8.得到输入X ，输出y
train_id = train['ID'].values
y = train['tradeMoney'].values.astype("float32")
print(y)
X = train[features].values
print("X shape:", X.shape)
print("y shape:", y.shape)

test_id = test['ID'].values
test_data = test[features].values
print("test shape", test_data.shape)

print("start：********************************")
start = time.time()

auc_list = []
pred_list = []

# 9.开始训练
# 采取分层采样
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


N = 5
skf = KFold(n_splits=N,shuffle=True,random_state=2018)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 参数设置
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eta': 0.02,
              # 'max_depth':4,
              # 'min_child_weight': 6,
              # 'colsample_bytree': 0.7,
              # 'subsample': 0.7,
              # 'eval_metric':'rmse',
              # 'gamma':0,
              # 'lambda':1,
              # 'alpha ':0，
              'silent': 1
              }
    # 数据结构
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtrain, 'train'), (dvali, 'valid')]  # 'valid-auc' will be used for early stopping
    # 模型train
    model = xgb.train(params, dtrain,
                      num_boost_round=2000,
                      evals=evallist,
                      early_stopping_rounds=500,
                      verbose_eval=100)
    # 预测验证
    pred = model.predict(dvali, ntree_limit=model.best_ntree_limit)
    # 评估
    auc = mean_squared_error(y_test, pred)
    print('...........................auc value:', auc)
    auc_list.append(auc)
    # 预测
    dtest = xgb.DMatrix(test_data)
    pre = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    pred_list.append(pre)

print('......................validate result mean :', np.mean(auc_list))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)

print("over:*********************************")

# 11.5折结果均值融合，并保存文件
mean_auc = np.mean(auc_list)
print("mean auc:", mean_auc)
filepath = 'output/xgb_' + str(mean_auc) + '.csv'  # 线下平均分数

# 转为array
res = np.array(pred_list)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = pd.DataFrame()
# result['ID'] = test_id
result['p'] = r
result.to_csv(filepath, header=False,index=False, sep=",")
