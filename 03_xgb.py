#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 03_xgb.py 
@time: 2019-04-21 23:56
@description:
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

train=pd.read_csv('input/train_data.csv')
test=pd.read_csv('input/test_a.csv')
print(len(train),len(test))
df=pd.concat([train,test],keys="ID",axis=0,sort=True)

no_features=['ID','tradeTime','tradeMoney','buildYear','region','plate','communityName']
categorical_feas=['rentType','houseType','houseFloor','houseToward','houseDecoration','city','region']
df=pd.get_dummies(df,columns=categorical_feas)
train,test=df[:len(train)],df[len(train):]
# train=pd.get_dummies(train,columns=categorical_feas)
# test=pd.get_dummies(test,columns=categorical_feas)
features=[fea for fea in train.columns if fea not in no_features]
print(features)
train.head().to_csv('demo.csv')
# 8.得到输入X ，输出y
train_id = train['ID'].values
y = train['tradeMoney'].values
print(y)
X = train[features].values
print("X shape:",X.shape)
print("y shape:",y.shape)

test_id = test['ID'].values
test_data = test[features].values
print("test shape",test_data.shape)

mean_re=[]
sub=pd.DataFrame()
k_fold = KFold(n_splits=5, shuffle=True, random_state=50)
for index, (train_index, test_index) in enumerate(k_fold.split(X)):
    # print(index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = xgb.XGBRegressor(objective='reg:linear', n_estimators=1000, min_child_weight=1,
                             learning_rate=0.01, max_depth=5, n_jobs=4,
                             subsample=0.6, colsample_bytree=0.4, colsample_bylevel=1)

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=30, verbose=2)
    # model = LinearRegression()
    # # print(y_train)
    # model.fit(X_train, y_train)

    vali_pre = model.predict(X_test)
    score = mean_absolute_error(y_test, vali_pre)
    mean_re.append(score)

    pred_result = model.predict(test_data)
    sub['p'] = pred_result
    if index == 0:
        re_sub = sub
    else:
        re_sub = re_sub + sub
re_sub = re_sub / 5
print(re_sub)
print('score list:', mean_re)
print(np.mean(mean_re))
re_sub.to_csv('output/03_xgb.csv', header=False,index=False, sep=",")