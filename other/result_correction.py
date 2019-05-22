#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: result_correction.py
@time: 2019-05-13 14:31
@description:
"""
import pandas as pd

# 训练集
df = pd.read_csv('../output/lgb_df.csv')
df = df[['area', 'rentType', 'houseType', 'houseFloor',
         'totalFloor', 'houseToward', 'houseDecoration', 'communityName',
         'tradeMoney']]
# 测试集
df_test = pd.read_csv('../input/test_a.csv')
df_pred = pd.read_csv('../output/lgb_0.941137595206218.csv', header=None)
df_test['pred'] = df_pred.values

# commnity_mean = dict()
# for index, group in df.groupby(by='communityName'):
#     commnity_mean[index] = group['tradeMoney'].mean()
#
# for index, group in df_test.groupby(by='communityName'):
#     if index not in commnity_mean:
#         commnity_mean[index] = group['pred'].mean()

# 根据规则进行调整
filter_rule_dict = dict()
df['filter_rule'] = df.apply(lambda row: str(row.area) +
                                         row.communityName +
                                         row.rentType +
                                         row.houseType +
                                         str(row.houseFloor) +
                                         str(row.totalFloor) +
                                         row.houseToward +
                                         row.houseDecoration,
                             axis=1)
df_test['filter_rule'] = df_test.apply(lambda row: str(row.area) +
                                                   row.communityName +
                                                   row.rentType +
                                                   row.houseType +
                                                   str(row.houseFloor) +
                                                   str(row.totalFloor) +
                                                   row.houseToward +
                                                   row.houseDecoration,
                                       axis=1)

for index, group in df.groupby(by='filter_rule'):
    filter_rule_dict[index] = group['tradeMoney'].mean()

# for index, group in df_test.groupby(by='filter_rule'):
#     if index not in filter_rule_dict:
#         filter_rule_dict[index] = group['pred'].mean()

df_test['mean'] = df_test.apply(
    lambda row: filter_rule_dict[row.filter_rule]
    if row.filter_rule in filter_rule_dict else row.pred,axis=1)

df_test['sub'] = df_test['pred'] * 0.5 + df_test['mean'] * 0.5
df_test['sub'].to_csv('result_correction.csv', index=None, header=None)
