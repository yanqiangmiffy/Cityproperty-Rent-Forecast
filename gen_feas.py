#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: gen_feas.py 
@time: 2019-04-22 18:26
@description:
"""
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv('input/train_data.csv')
df_test = pd.read_csv('input/test_a.csv')
df = pd.concat([df_train, df_test], sort=False, axis=0, ignore_index=True)

# 数据预处理

df['rentType'] = df['rentType'].replace('--', '未知方式')
house_type_nums = df['houseType'].value_counts().to_dict()


# 将房屋类型计数分箱
def check_type(x):
    if house_type_nums[x] >= 1000:
        return "high_num"
    elif 100 <= house_type_nums[x] < 1000:
        return "median_num"
    else:
        return "low_num"


df['houseType'] = df['houseType'].apply(lambda x: check_type(x))

# 交易至今的天数
now = datetime.datetime.now()
df['tradeTime'] = pd.to_datetime(df['tradeTime'])
df['interval'] = (now - df['tradeTime']).dt.days

# 缺失值处理
df['pv'] = df['pv'].fillna(value=df['pv'].median())
df['uv'] = df['uv'].fillna(value=df['uv'].median())


# 类别特征 具有大小关系编码
categorical_feas = ['rentType', 'houseType', 'houseFloor', 'region', 'plate', 'houseToward', 'houseDecoration']

for col in categorical_feas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df['stationNum'] = df['subwayStationNum'] + df['busStationNum']
df['schoolNum'] = df['interSchoolNum'] + df['schoolNum'] + df['privateSchoolNum']
df['medicalNum'] = df['hospitalNum'] + df['drugStoreNum']
df['lifeHouseNum'] = df['gymNum'] + df['bankNum'] + df['shopNum'] + df['parkNum'] + df['mallNum'] + df['superMarketNum']
df['landSupplyTradeRatio'] = df['supplyLandArea'] / df['tradeLandArea']

df = df.drop(['subwayStationNum', 'busStationNum',
              'interSchoolNum', 'schoolNum', 'privateSchoolNum',
              'hospitalNum', 'drugStoreNum',
              'gymNum', 'bankNum', 'shopNum', 'parkNum', 'mallNum', 'superMarketNum',
              'supplyLandArea', 'tradeLandArea'], axis=1)
df.to_csv('input/df.csv', index=False)


# 特征工程
no_features = ['ID', 'tradeTime', 'tradeMoney',
               'buildYear', 'communityName', 'city',
               '']
features = [fea for fea in df.columns if fea not in no_features]

train, test = df[:len(df_train)], df[len(df_train):]
print(train.shape, test.shape)


def load_data():
    return train, test, no_features, features
