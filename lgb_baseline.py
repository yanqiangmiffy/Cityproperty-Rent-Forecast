#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: lgb_baseline.py 
@time: 2019-04-23 15:18
@description: lgb 模型
"""

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from gen_feas import load_data

train,test,no_features,features=load_data()
print(train.head())
print(len(features))