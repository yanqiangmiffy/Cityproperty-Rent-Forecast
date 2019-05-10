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
    grouped_df.columns = ['communityName_'+'_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    print(grouped_df)

    df = pd.merge(df, grouped_df, on='communityName', how='left')

df.head(100).to_csv('input/df_sample.csv', index=False)
