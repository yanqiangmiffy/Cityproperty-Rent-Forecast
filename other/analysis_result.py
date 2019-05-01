#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: analysis_result.py 
@time: 2019-04-26 16:44
@description: 结果分析 从交叉验证上面分析误差
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [u'Microsoft YaHei']

true_pred = pd.read_csv('output/full_df.csv')
true_pred['error'] = abs(true_pred['tradeMoney'] - true_pred['pred_tradeMoney'])
# print(true_pred['error'])
print(true_pred[true_pred['error'] >= 10000])

# 分析训练街和测试集
## area
train = pd.read_csv('input/train_data.csv')
test = pd.read_csv('input/test_a.csv')
print(train['area'].describe())
print(test['area'].describe())

# 训练集area最大值15055 最小值为1
# 测试集area最大值为150 最小值为15

## houseType
print("houseType")
print(train['houseType'].value_counts())
print(test['houseType'].value_counts(), len(test['houseType'].unique()))

test['houseType'].value_counts().plot(kind="bar")
plt.show()

# houseFloor
print("houseFloor")
print(train['houseFloor'].value_counts())
print(test['houseFloor'].value_counts())

# totalFloor
print("totalFloor")
print(train['totalFloor'].describe())
print(test['totalFloor'].describe())

# train totalFloor 最小值 0 最大值 88
# test  totalFloor 最小值 2 最大值 53

# communityName
print("communityName")
print(train['communityName'].value_counts())
print(test['communityName'].value_counts())

# region
print("region")
print(train['region'].value_counts(), len(train['region'].value_counts()))
print(test['region'].value_counts(), len(test['region'].value_counts()))

# plate
print("plate")
print(train['plate'].value_counts(), len(train['plate'].value_counts()))
print(test['plate'].value_counts(), len(test['plate'].value_counts()))
