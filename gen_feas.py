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
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from itertools import combinations
from sklearn.cluster import KMeans

df_train = pd.read_csv('input/train_data.csv')
df_test = pd.read_csv('input/test_a.csv')

print("filter tradeMoney before:", len(df_train))
df_train = df_train.query("500<=tradeMoney<20000")  # 线下 lgb_0.876612870005764
print("filter tradeMoney after:", len(df_train))

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
df['mean_area'] = df['area'] / df['house_total_num']

# 交易至今的天数
df['交易月份'] = df['tradeTime'].apply(lambda x: int(x.split('/')[1]))
now = datetime.now()
df['tradeTime'] = pd.to_datetime(df['tradeTime'])
df['now_trade_interval'] = (now - df['tradeTime']).dt.days

# 我们使用get_dummies()进行编码或者label
df['tradeTime_month'] = df['tradeTime'].dt.month
# [(month % 12 + 3) // 3 for month in range(1, 13)]
df['tradeTime_season'] = df['tradeTime_month'].apply(lambda month: (month % 12 + 3) // 3)

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

# 其他特征
df['stationNum'] = df['subwayStationNum'] + df['busStationNum']
df['schoolNum'] = df['interSchoolNum'] + df['schoolNum'] + df['privateSchoolNum']
df['medicalNum'] = df['hospitalNum'] + df['drugStoreNum']
df['lifeHouseNum'] = df['gymNum'] + df['bankNum'] + df['shopNum'] + df['parkNum'] + df['mallNum'] + df['superMarketNum']

# 重要特征
df['area_floor_ratio'] = df['area'] / (df['totalFloor'] + 1)
df['uv_pv_ratio'] = df['uv'] / (df['pv'] + 1)
df['uv_pv_sum'] = df['uv'] + df['pv']

# --------- 小区特征 -----------
# 每个小区交易次数
community_trade_nums = dict(df['communityName'].value_counts())
df['community_nums'] = df['communityName'].apply(lambda x: community_trade_nums[x])

# 每个小区的特征最小值、最大值、平均值,求和、中位数
community_feas = ['area', 'mean_area', 'now_trade_interval',
                  'now_build_interval', 'totalFloor',
                  'tradeMeanPrice', 'tradeNewMeanPrice',
                  'totalTradeMoney', 'totalTradeArea', 'remainNewNum',
                  'uv_pv_ratio', 'pv', 'uv', '交易月份','lookNum'
                  ]
numerical_feas = ['area', 'totalFloor', 'saleSecHouseNum', 'subwayStationNum',
                  'busStationNum', 'interSchoolNum', 'schoolNum', 'privateSchoolNum', 'hospitalNum',
                  'drugStoreNum', 'gymNum', 'bankNum', 'shopNum', 'parkNum', 'mallNum', 'superMarketNum',
                  'totalTradeMoney', 'totalTradeArea', 'tradeMeanPrice', 'tradeSecNum', 'totalNewTradeMoney',
                  'totalNewTradeArea', 'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum', 'supplyNewNum',
                  'supplyLandNum', 'supplyLandArea', 'tradeLandNum', 'tradeLandArea', 'landTotalPrice',
                  'landMeanPrice', 'totalWorkers', 'newWorkers', 'residentPopulation', 'pv', 'uv', 'lookNum']

for fea in tqdm(set(community_feas + numerical_feas)):
    grouped_df = df.groupby('communityName').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
    grouped_df.columns = ['communityName_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on='communityName', how='left')

# --------- 版块特征 -----------
# 每个板块交易次数
plate_trade_nums = dict(df['plate'].value_counts())
df['plate_nums'] = df['plate'].apply(lambda x: plate_trade_nums[x])

for fea in tqdm(community_feas):
    grouped_df = df.groupby('plate').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
    grouped_df.columns = ['plate_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)

    df = pd.merge(df, grouped_df, on='plate', how='left')

# ----------- 地区特征 -------------
# region_trade_nums = dict(df['region'].value_counts())
# df['region_nums'] = df['region'].apply(lambda x: region_trade_nums[x])
#
# for fea in community_feas:
#     grouped_df = df.groupby('region').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
#     grouped_df.columns = ['region_' + '_'.join(col).strip() for col in grouped_df.columns.values]
#     grouped_df = grouped_df.reset_index()
#     # print(grouped_df)
#
#     df = pd.merge(df, grouped_df, on='region', how='left')

# # 月份特征
# tradeTime_month_nums = dict(df['tradeTime_month'].value_counts())
# df['tradeTime_month_nums'] = df['tradeTime_month'].apply(lambda x: tradeTime_month_nums[x])
#
# for fea in community_feas:
#     grouped_df = df.groupby('tradeTime_month').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
#     grouped_df.columns = ['tradeTime_month_' + '_'.join(col).strip() for col in grouped_df.columns.values]
#     grouped_df = grouped_df.reset_index()
#     # print(grouped_df)
#     df = pd.merge(df, grouped_df, on='tradeTime_month', how='left')
# # 季节特征
#
# tradeTime_season_nums = dict(df['tradeTime_season'].value_counts())
# df['tradeTime_season_nums'] = df['tradeTime_season'].apply(lambda x: tradeTime_month_nums[x])
#
# for fea in community_feas:
#     grouped_df = df.groupby('tradeTime_season').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
#     grouped_df.columns = ['tradeTime_season_' + '_'.join(col).strip() for col in grouped_df.columns.values]
#     grouped_df = grouped_df.reset_index()
#     # print(grouped_df)
#     df = pd.merge(df, grouped_df, on='tradeTime_season', how='left')

# ----------- 类别特征 具有大小关系编码 -------------
# com_categorical_feas = ['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration']
#
# for col in com_categorical_feas:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     # df[col] = df[col].astype('category')
df = pd.get_dummies(df, columns=['tradeTime_month', 'tradeTime_season'])
categorical_feas = ['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration', 'region', 'plate']
df = pd.get_dummies(df, columns=categorical_feas)

# 生成数据
no_features = ['ID', 'tradeTime', 'tradeMoney',
               'buildYear', 'communityName', 'city', 'area_money'
               ]

# no_features = no_features + too_many_zeros
no_features = no_features

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(df_train)], df[len(df_train):]

print("训练集和测试集维度：", train.shape, test.shape)
df.head(100).to_csv('input/df.csv', index=False)
print(features)


def load_data():
    return train, test, no_features, features
