#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: gen_feas4.py
@time: 2019-04-22 18:26
@description:
"""
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from itertools import combinations
from sklearn.cluster import KMeans

df_train = pd.read_csv('input/train_data.csv')
df_test = pd.read_csv('input/test_a.csv')

print("filter tradeMoney before:", len(df_train))
# df_train = df_train.query("tradeMoney>=1000&tradeMoney<15000") # 线下0.87 线上0.86
# df_train = df_train.query("tradeMoney>=500&tradeMoney<40000")#线下0.8816  线上0.84
# df_train = df_train.query("tradeMoney>=500&tradeMoney<25000")# 线下 0.8857
# df_train = df_train.query("tradeMoney>=500&tradeMoney<20000") # 线下 0.8836
# df_train = df_train.query("tradeMoney>=500&tradeMoney<18000") # 线下 0.867
# df_train = df_train.query("tradeMoney>=800&tradeMoney<16000") # 线下 lgb_0.8757434770663066
df_train = df_train.query("500<=tradeMoney<20000")  # 线下 lgb_0.876612870005764

print("filter area after:", len(df_train))
df_train = df_train.query("15<=area<=150")  # 线下 lgb_0.8830538988139025 线上0.867
print("filter area after:", len(df_train))

df_train['area_money'] = df_train['tradeMoney'] / df_train['area']
df_train = df_train.query("15<=area_money<300")  # 线下 lgb_0.9003567192921244.csv 线上0.867649
print("filter area/money after:", len(df_train))

#
# totalFloor
# print("filter totalFloor after:", len(df_train))
# df_train = df_train.query("2<=totalFloor<=53")
# print("filter totalFloor after:", len(df_train))
#
# unique_comname = df_test['communityName'].unique()
# print("filter communityName after:", len(df_train))
# df_train = df_train[df_train['communityName'].isin(unique_comname)]
# print("filter communityName after:", len(df_train))
#
# print("houseType")
#
# unique_house = df_test['houseType'].unique()
# print("filter houseType after:", len(df_train))
# df_train = df_train[df_train['houseType'].isin(unique_house)]
# print("filter houseType after:", len(df_train))


df = pd.concat([df_train, df_test], sort=False, axis=0, ignore_index=True)

# 数据预处理
df['rentType'] = df['rentType'].replace('--', '未知方式')
house_type_nums = df['houseType'].value_counts().to_dict()


def split_type(x):
    """
    分割房屋类型
    :param x:
    :return:
    """
    assert len(x) == 6, "x的长度必须为6"
    return int(x[0]), int(x[2]), int(x[4])


df['houseType_shi'], df['houseType_ting'], df['houseType_wei'] = zip(*df['houseType'].apply(lambda x: split_type(x)))
df['house_total_num'] = df['houseType_shi'] + df['houseType_ting'] + df['houseType_wei']

# def check_type(x):
#     """
#     将房屋类型计数分箱
#     :param x:
#     :return:
#     """
#     if house_type_nums[x] >= 1000:
#         return "high_num"
#     elif 100 <= house_type_nums[x] < 1000:
#         return "median_num"
#     else:
#         return "low_num"
#
#
# df['houseType'] = df['houseType'].apply(lambda x: check_type(x))

# 交易至今的天数
df['交易月份'] = df['tradeTime'].apply(lambda x: int(x.split('/')[1]))
now = datetime.now()
df['tradeTime'] = pd.to_datetime(df['tradeTime'])
df['now_trade_interval'] = (now - df['tradeTime']).dt.days


# 我们使用get_dummies()进行编码或者label
df['tradeTime_month'] = df['tradeTime'].dt.month
# [(month % 12 + 3) // 3 for month in range(1, 13)]
df['tradeTime_season'] = df['tradeTime_month'].apply(lambda month: (month % 12 + 3) // 3)
df = pd.get_dummies(df, columns=['tradeTime_month', 'tradeTime_season'])

df['buildYear'] = df['buildYear'].replace('暂无信息', 0)
df['buildYear'] = df['buildYear'].astype(int)
# 直接使用小区的构建年份填充暂无信息
# df['buildYear_allmean'] = df['buildYear'].replace(0, int(df['buildYear'].mean()))

buildyear_median = dict()
buildyear_mean = dict()
buildyear_mode = dict()
for index, group in df[['region', 'buildYear']].groupby("region"):
    buildyear_median[index] = int(group['buildYear'].median())
    buildyear_mean[index] = int(group['buildYear'].median())
    buildyear_mode[index] = int(group['buildYear'].median())
    # print(index,group['buildYear'].median())
    # print(index,group['buildYear'].mean())
    # print(index,group['buildYear'].mode())


def replace_zero(row):
    # if row.buildYear == 0:
    #     return buildyear_median[row.region]
    # if row.buildYear == 0:
    #     return buildyear_mean[row.region]
    if row.buildYear == 0:
        return buildyear_mode[row.region]
    else:
        return row.buildYear


# 使用相同板块内的中位数年份、平均数年份、众数年份
df['buildYear'] = df.apply(lambda row: replace_zero(row), axis=1)
df['now_build_interval'] = 2019 - df['buildYear']

# 缺失值处理
df['pv'] = df['pv'].fillna(value=int(df['pv'].median()))
df['uv'] = df['uv'].fillna(value=int(df['uv'].median()))

# 类别特征 具有大小关系编码
# com_categorical_feas = ['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration']
#
# for col in com_categorical_feas:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     # df[col] = df[col].astype('category')
categorical_feas = ['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration', 'region', 'plate']
df = pd.get_dummies(df, columns=categorical_feas)


# 其他特征
df['stationNum'] = df['subwayStationNum'] + df['busStationNum']
df['schoolNum'] = df['interSchoolNum'] + df['schoolNum'] + df['privateSchoolNum']
df['medicalNum'] = df['hospitalNum'] + df['drugStoreNum']
df['lifeHouseNum'] = df['gymNum'] + df['bankNum'] + df['shopNum'] + df['parkNum'] + df['mallNum'] + df['superMarketNum']

# 重要特征
df['area_floor_ratio'] = df['area'] / (df['totalFloor'] + 1)
df['uv_pv_ratio'] = df['uv'] / df['pv'] + 1
df['uv_pv_sum'] = df['uv'] + df['pv']

# 特征工程
no_features = ['ID', 'tradeTime', 'tradeMoney',
               'buildYear', 'communityName', 'city', 'area_money'
               ]

# no_features = no_features + too_many_zeros
no_features = no_features

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(df_train)], df[len(df_train):]

print(train.shape, test.shape)
df.to_csv('input/df.csv', index=False)
print(features)


def load_data():
    return train, test, no_features, features
