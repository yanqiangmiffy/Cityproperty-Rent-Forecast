#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: analysis_result.py.py
@time: 2019-05-16 23:19
@description:
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [u'Microsoft YaHei']

# true_pred = pd.read_csv('../output/lgb_df.csv')
# true_pred['error'] = abs(true_pred['tradeMoney'] - true_pred['pred_tradeMoney'])
# # print(true_pred['error'])
# # print(true_pred[true_pred['error'] >= 10000])
# print(true_pred.describe())
# print(true_pred.describe().columns)
# # 分析训练街和测试集
# ## area
train = pd.read_csv('../input/train_data.csv')
test = pd.read_csv('../input/test_a.csv')
df = pd.concat([train, test], ignore_index=True, sort=False)
# print(df)

print(df['houseToward'].value_counts())
print(df['rentType'].value_counts())
print(df['houseDecoration'].value_counts())

community_dec = dict()
for index, group in df.groupby(['communityName']):
    decs = group['houseDecoration'].value_counts().index.tolist()
    if decs[0] == '其他' and len(decs) == 1:
        community_dec[index] = decs[0]
    else:
        if len(decs) == 1:
            community_dec[index] = decs[0]
        else:
            community_dec[index] = decs[1]

df['houseDecoration'] = df.apply(
    lambda x: community_dec[x.communityName] if x.houseDecoration == '其他' else x.houseDecoration, axis=1)
print(df['houseDecoration'].value_counts())
# grouped_df = df.groupby(['communityName', 'rentType']).agg({'area': ['min', 'max', 'mean', 'sum', 'median']})
# print(grouped_df)
#
# grouped_df.columns = ['communityName_' + '_'.join(col).strip() for col in grouped_df.columns.values]
# grouped_df = grouped_df.reset_index()
# print(grouped_df.columns)
# print(train['area'].describe())
# print(test['area'].describe())
#
# # 训练集area最大值15055 最小值为1
# # 测试集area最大值为150 最小值为15
#
# ## houseType
# print("houseType")
# print(train['houseType'].value_counts())
# print(test['houseType'].value_counts(), len(test['houseType'].unique()))
#
# test['houseType'].value_counts().plot(kind="bar")
# plt.show()
#
# # houseFloor
# print("houseFloor")
# print(train['houseFloor'].value_counts())
# print(test['houseFloor'].value_counts())
#
# # totalFloor
# print("totalFloor")
# print(train['totalFloor'].describe())
# print(test['totalFloor'].describe())

# train totalFloor 最小值 0 最大值 88
# test  totalFloor 最小值 2 最大值 53

# # communityName
# print("communityName")
# print(train['communityName'].value_counts())
# print(test['communityName'].value_counts())
#
# # region
# print("region")
# print(train['region'].value_counts(), len(train['region'].value_counts()))
# print(test['region'].value_counts(), len(test['region'].value_counts()))
#
# # plate
# print("plate")
# print(train['plate'].value_counts(), len(train['plate'].value_counts()))
# print(test['plate'].value_counts(), len(test['plate'].value_counts()))

#
# for fea in tqdm(community_feas):
#     an_df= df.groupby(['communityName', 'rentType']).agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
#     an_df.columns = ['communityName_rentType_' + '_'.join(col).strip() for col in an_df.columns.values]
#     an_df = an_df.reset_index()
#     # print(grouped_df)
#
#     df = pd.merge(df, an_df, on=['communityName','rentType'], how='left')
