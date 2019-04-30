#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_xgb.py 
@time: 2019-04-28 23:01
@description:
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from scipy.stats import skew
from collections import OrderedDict
from gen_feas import load_data
from utils import my_score

train, test, no_features, features = load_data()
X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values

xg_reg = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.07,
                 max_depth=3,
                 # min_child_weight=1.5,
                 n_estimators=2000,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)

xg_reg.fit(X,y,verbose=True)


lasso = Lasso(alpha =0.0005, random_state=1)
lasso.fit(X,y)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
GBoost.fit(X,y)



YpredictedonTrain = xg_reg.predict(X)
Ypredicted2onTrain = GBoost.predict(X)
Ypredicted3onTrain = lasso.predict(X)
print(YpredictedonTrain.shape)
print(Ypredicted2onTrain.shape)
print(Ypredicted3onTrain.shape)

dfinal = pd.DataFrame({'a':YpredictedonTrain, 'b':Ypredicted2onTrain,'c':Ypredicted3onTrain})
dfinal.head()
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(dfinal,y)




a = xg_reg.predict(test_data)
b = GBoost.predict(test_data)
c = lasso.predict(test_data)
dStack = pd.DataFrame({'a':a, 'b':b,'c':c})
dStack.head()



output = model_lgb.predict(dStack)
df = pd.DataFrame(output)
df.head()
df.to_csv('output2.csv',index=False)


