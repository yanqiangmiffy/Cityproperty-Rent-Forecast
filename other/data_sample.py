#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: data_sample.py
@time: 2019-05-09 22:36
@description:
"""

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from itertools import combinations
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', 100)
df_train = pd.read_csv('input/train_data.csv')
df_test = pd.read_csv('input/test_a.csv')
print("filter tradeMoney before:", len(df_train))
df_train = df_train.query("900<=tradeMoney<16000")  # 线下 lgb_0.876612870005764
print("filter tradeMoney after:", len(df_train))

df_train = df_train.query("15<=area<=150")  # 线下 lgb_0.8830538988139025 线上0.867
print("filter area after:", len(df_train))

df_train['area_money'] = df_train['tradeMoney'] / df_train['area']
df_train = df_train.query("15<=area_money<300")  # 线下 lgb_0.9003567192921244.csv 线上0.867649
print("filter area/money after:", len(df_train))

df = pd.concat([df_train, df_test], sort=False, axis=0, ignore_index=True)

feas = ['area']
for fea in feas:
    grouped_df = df.groupby('communityName').agg({fea: ['min', 'max', 'mean']})
    grouped_df.columns = ['communityName_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)

    df = pd.merge(df, grouped_df, on='communityName', how='left')
    for col in grouped_df:
        if col != 'communityName':
            df[fea + '&' + col] = df[fea] - df[col]
            df[fea + '/' + col] = df[fea] / (1 + df[col])
            df[fea + '*' + col] = df[fea] * df[col]
            df[fea + '+' + col] = df[fea] + df[col]

df.head(100).to_csv('input/df_sample.csv', index=False)

rank_df = df_train.loc[:, ['houseToward', 'tradeMoney']].groupby('houseToward', as_index=False).mean().sort_values(
    by='tradeMoney').reset_index(drop=True)
rank_df.loc[:, 'houseToward' + '_rank'] = rank_df.index + 1
rank_fe_df = rank_df.drop(['tradeMoney'], axis=1)
df_train = df_train.merge(rank_fe_df, how='left', on='houseToward')  ###划重点！！！！
df_test = df_test.merge(rank_fe_df, how='left', on='houseToward')

rank_cols = ['area', 'totalFloor', 'saleSecHouseNum', 'subwayStationNum',
                  'busStationNum', 'interSchoolNum', 'schoolNum', 'privateSchoolNum', 'hospitalNum',
                  'drugStoreNum', 'gymNum', 'bankNum', 'shopNum', 'parkNum', 'mallNum', 'superMarketNum',
                  'totalTradeMoney', 'totalTradeArea', 'tradeMeanPrice', 'tradeSecNum', 'totalNewTradeMoney',
                  'totalNewTradeArea', 'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum', 'supplyNewNum',
                  'supplyLandNum', 'supplyLandArea', 'tradeLandNum', 'tradeLandArea', 'landTotalPrice',
                  'landMeanPrice', 'totalWorkers', 'newWorkers', 'residentPopulation', 'pv', 'uv', 'lookNum']
for col in rank_cols:
    if col != 'tradeMoney':
        print(col + '_rank_encoding...')
        tmp_train_df = df_train.copy()
        tmp_val_df = df_test.copy()

        rank_df = df_train.loc[:, [col, 'tradeMoney']].groupby(col, as_index=False).mean().sort_values(
            by='tradeMoney').reset_index(drop=True)
        rank_df.loc[:, col + '_rank'] = rank_df.index + 1  # +1，为缺失值预留一个0值的rank
        rank_fe_df = rank_df.drop(['tradeMoney'], axis=1)
        df_train = tmp_train_df.merge(rank_fe_df, how='left', on=col)
        df_test = tmp_val_df.merge(rank_fe_df, how='left', on=col)
        # tmp_train_df.drop([col],axis=1,inplace=True)
        # tmp_val_df.drop([col],axis=1,inplace=True)
print(df_train,df_test)