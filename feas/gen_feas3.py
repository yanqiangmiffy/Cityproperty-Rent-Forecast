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

# df['tradeTime']=pd.to_datetime(df['tradeTime'])
# df['interval']=(now-df['tradeTime']).dt.days
# no_features = ['ID', 'tradeTime', 'tradeMoney',]
df_train = pd.read_csv('input/train_data.csv')
print("filter before:", len(df_train))
# df_train = df_train.query("tradeMoney>=1000&tradeMoney<15000") # 线下0.87 线上0.86
# df_train = df_train.query("tradeMoney>=500&tradeMoney<40000")#线下0.8816  线上0.84
# df_train = df_train.query("tradeMoney>=500&tradeMoney<25000")# 线下 0.8857
# df_train = df_train.query("tradeMoney>=500&tradeMoney<20000") # 线下 0.8836
# df_train = df_train.query("tradeMoney>=500&tradeMoney<18000") # 线下 0.867
# df_train = df_train.query("tradeMoney>=800&tradeMoney<16000") # 线下 lgb_0.8757434770663066
df_train = df_train.query("tradeMoney>=900&tradeMoney<16000")  # 线下 lgb_0.876612870005764

print("filter after:", len(df_train))

df_test = pd.read_csv('input/test_a.csv')
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


def check_type(x):
    """
    将房屋类型计数分箱
    :param x:
    :return:
    """
    if house_type_nums[x] >= 1000:
        return "high_num"
    elif 100 <= house_type_nums[x] < 1000:
        return "median_num"
    else:
        return "low_num"


df['houseType'] = df['houseType'].apply(lambda x: check_type(x))

# 交易至今的天数
now = datetime.now()
df['tradeTime'] = pd.to_datetime(df['tradeTime'])
df['now_trade_interval'] = (now - df['tradeTime']).dt.days

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
categorical_feas = ['rentType', 'houseType', 'houseFloor', 'region', 'plate', 'houseToward', 'houseDecoration']

for col in categorical_feas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # df[col] = df[col].astype('category')

# # 添加组合特征
# relate_pairs1 = ['saleSecHouseNum', 'subwayStationNum', 'busStationNum', 'interSchoolNum', 'schoolNum',
#                  'privateSchoolNum', 'hospitalNum', 'drugStoreNum', 'gymNum', 'bankNum', 'shopNum', 'parkNum',
#                  'mallNum', 'superMarketNum', 'tradeSecNum', 'tradeNewNum', 'remainNewNum',
#                  'supplyNewNum', 'totalWorkers', 'newWorkers', 'residentPopulation']
# for rv in combinations(relate_pairs1, 2):
#     rv2 = '_'.join(rv)
#     df[rv2 + '_sum'] = df[rv[0]] + df[rv[1]]
#     df[rv2 + '_diff'] = df[rv[0]] - df[rv[1]]
#     df[rv2 + '_multiply'] = df[rv[0]] * df[rv[1]]
#     df[rv2 + '_div'] = df[rv[0]] / (df[rv[1]] + 1)
#
# relate_pairs2 = ['totalTradeArea', 'tradeMeanPrice', 'totalTradeMoney', 'tradeNewMeanPrice', 'totalNewTradeArea',
#                  'totalNewTradeMoney', 'area']
# for rv in combinations(relate_pairs2, 2):
#     rv2 = '_'.join(rv)
#     df[rv2 + '_sum'] = df[rv[0]] + df[rv[1]]
#     df[rv2 + '_diff'] = df[rv[0]] - df[rv[1]]
#     df[rv2 + '_multiply'] = df[rv[0]] * df[rv[1]]
#     df[rv2 + '_div'] = df[rv[0]] / (df[rv[1]] + 1)
#
# relate_pairs3 = ['pv', 'uv', 'lookNum']
# for rv in combinations(relate_pairs3, 2):
#     rv2 = '_'.join(rv)
#     df[rv2 + '_sum'] = df[rv[0]] + df[rv[1]]
#     df[rv2 + '_diff'] = df[rv[0]] - df[rv[1]]
#     df[rv2 + '_multiply'] = df[rv[0]] * df[rv[1]]
#     df[rv2 + '_div'] = df[rv[0]] / (df[rv[1]] + 1)
#
# too_many_zeros = ['supplyLandNum', 'supplyLandArea', 'tradeLandNum',
#                   'tradeLandArea', 'landTotalPrice', 'landMeanPrice']
# for rv in combinations(too_many_zeros, 2):
#     rv2 = '_'.join(rv)
#     df[rv2 + '_sum'] = df[rv[0]] + df[rv[1]]
#     df[rv2 + '_diff'] = df[rv[0]] - df[rv[1]]
#     df[rv2 + '_multiply'] = df[rv[0]] * df[rv[1]]
#     df[rv2 + '_div'] = df[rv[0]] / (df[rv[1]] + 1)

# 其他特征
df['stationNum'] = df['subwayStationNum'] + df['busStationNum']
df['schoolNum'] = df['interSchoolNum'] + df['schoolNum'] + df['privateSchoolNum']
df['medicalNum'] = df['hospitalNum'] + df['drugStoreNum']
df['lifeHouseNum'] = df['gymNum'] + df['bankNum'] + df['shopNum'] + df['parkNum'] + df['mallNum'] + df['superMarketNum']
df['landSupplyTradeRatio'] = df['supplyLandArea'] / df['tradeLandArea']
# 对数值型的特征，处理为rank特征（鲁棒性好一点）
# numerical_features=df.loc[:,'saleSecHouseNum':'now_build_interval'].columns.tolist()+['area','totalFloor']
# for feat in numerical_features:
#     df[feat] = df[feat].rank() / float(df.shape[0]) # 排序，并且进行归一化
# rank_feas=['totalTradeMoney']
# df['totalTradeMoney']=df['totalTradeMoney'].rank()
# print(df['totalTradeMoney'])

# 重要特征
df['area_floor_ratio']=df['area']/df['totalFloor']
df['tradeMeanPrice_new_ratio']=df['tradeMeanPrice']/df['tradeNewMeanPrice']
df['tradeMeanPrice_new_sum']=df['tradeMeanPrice']+df['tradeNewMeanPrice']+df['totalTradeMoney']
df['uv_pv_ratio']=df['uv']/df['pv']
df['uv_pv_sum']=df['uv']+df['pv']

# 特征工程
no_features = ['ID', 'tradeTime', 'tradeMoney',
               'buildYear', 'communityName', 'city'
               ]

# no_features = no_features + too_many_zeros
no_features = no_features

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(df_train)], df[len(df_train):]

print(train.shape, test.shape)
df[:20].to_csv('input/df.csv', index=False)
print(features)


def load_data():
    return train, test, no_features, features
