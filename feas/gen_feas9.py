#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: gen_feas4.py
@time: 2019-04-22 18:26
@description:
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from itertools import combinations
from sklearn.cluster import KMeans
from utils import numerical_feas

df_train = pd.read_csv('input/train_data.csv')
df_test = pd.read_csv('input/test_a.csv')

# 加载word2vec feat
# df_train = pd.read_csv('input/train_w2v.csv')
# df_test = pd.read_csv('input/test_w2v.csv')


# ------------------ 过滤数据 begin ----------------
print("根据tradeMoney过滤数据:", len(df_train))
df_train = df_train.query("400<=tradeMoney<30000")
print("filter tradeMoney after:", len(df_train))

print("根据area过滤数据:", len(df_train))
df_train = df_train.query("0<area<=170")
print("filter area after:", len(df_train))

print("根据tradeMoney/area过滤数据:", len(df_train))
df_train['area_money'] = df_train['tradeMoney'] / df_train['area']
df_train = df_train.query("20<=area_money<250")
print("filter area/money after:", len(df_train))
#
# with open('log.txt', 'a', encoding='utf-8') as f:
#     print('-' * 100 + '\n')
#     f.write('-' * 50 + '\n')
#     f.write("根据上次训练的结果，过滤误差较大的数据：\n")
#     lgb_df = pd.read_csv('output/lgb_df.csv')
#     lgb_df['error'] = abs(lgb_df['tradeMoney'] - lgb_df['pred_tradeMoney'])
#     f.write("误差大的数据个数" + str(len(lgb_df.query('error>=5000'))) + "\n")
#     lgb_df = lgb_df.query('error<=5000')
#     small_error_ids = lgb_df.ID.values
#     df_train = df_train[df_train['ID'].isin(small_error_ids)]
#     f.write("过滤误差大的训练集之后数据个数：" + str(len(df_train)) + "\n")

#
# totalFloor
# print("filter totalFloor after:", len(df_train))
# df_train = df_train.query("2<=totalFloor<=53")
# print("filter totalFloor after:", len(df_train))

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
# ------------------ 过滤数据 end ----------------

df = pd.concat([df_train, df_test], sort=False, axis=0, ignore_index=True)

# 数据预处理
df['rentType'] = df['rentType'].replace('--', '未知方式')
# rentType租房形式处理，可以根据租房面积将"未知方式"替换为"整租"或者"合租"
# 整租租房面积大，合租租房面积小
rentType_by_area = dict()
for index, group in df.groupby(by="rentType"):
    rentType_by_area[index] = group['area'].mean()


def replace_renttype(row):
    if row.rentType == '未知方式':
        if row.area >= rentType_by_area['整租']:
            return '整租'
        else:
            return '合租'
    else:
        return row.rentType


df['rentType'] = df.apply(lambda row: replace_renttype(row), axis=1)
print(df['rentType'].value_counts())


# 装修方式
# community_dec = dict()
# for index, group in df.groupby(['communityName']):
#     decs = group['houseDecoration'].value_counts().index.tolist()
#     if decs[0] == '其他' and len(decs) == 1:
#         community_dec[index] = decs[0]
#     else:
#         if len(decs) == 1:
#             community_dec[index] = decs[0]
#         else:
#             community_dec[index] = decs[1]
#
# df['houseDecoration'] = df.apply(
#     lambda x: community_dec[x.communityName] if x.houseDecoration == '其他' else x.houseDecoration, axis=1)


def split_type(x):
    """
    分割房屋类型
    :param x:
    :return:
    """
    assert len(x) == 6, "x的长度必须为6"
    return int(x[0]), int(x[2]), int(x[4])


# -----房屋面积、卧室数量、厅的数量、卫的数量进行特征提取 begin ------

df['室数量'], df['厅数量'], df['卫数量'] = zip(*df['houseType'].apply(lambda x: split_type(x)))
df['室厅数量'] = df['室数量'] + df['厅数量']
df['室卫数量'] = df['室数量'] + df['卫数量']
df['厅卫数量'] = df['厅数量'] + df['卫数量']
df['室卫厅数量'] = df['室数量'] + df['厅数量'] + df['卫数量']
df['mean_area'] = df['area'] / df['室卫厅数量']

df['室占比'] = df['室数量'] / df['室卫厅数量']
df['厅占比'] = df['厅数量'] / df['室卫厅数量']
df['卫占比'] = df['卫数量'] / df['室卫厅数量']

df['室面积'] = df['area'] * df['室占比']
df['厅面积'] = df['area'] * df['厅占比']
df['卫面积'] = df['area'] * df['卫占比']


# -----房屋面积、卧室数量、厅的数量、卫的数量进行特征提取 end ------

# ------ 房屋楼层特征 begin -------
def house_floor(x):
    if x == '低':
        r = 0
    elif x == '中':
        r = 0.3333
    else:
        r = 0.6666
    return r


df['houseFloor_ratio'] = df['houseFloor'].apply(lambda x: house_floor(x))
df['所在楼层'] = df['totalFloor'] * df['houseFloor_ratio']
# ------ 房屋楼层特征 end -------

# ------- 名字数字 begin ---------
df['小区名字的数字'] = df['communityName'].apply(lambda x: int(x.replace('XQ', '')))
df['板块名字的数字'] = df['plate'].apply(lambda x: int(x.replace('BK', '')))

# 交易至今的天数
df['交易月份'] = df['tradeTime'].apply(lambda x: int(x.split('/')[1]))
# now = datetime.now()
now = datetime.strptime('2019-04-27', '%Y-%m-%d')  # 5-11
df['tradeTime'] = pd.to_datetime(df['tradeTime'])
df['now_trade_interval'] = (now - df['tradeTime']).dt.days
end_2018 = datetime.strptime('2018-12-31', '%Y-%m-%d')
df['2018_trade_interval'] = (end_2018 - df['tradeTime']).dt.days

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
# 每个小区的特征最小值、最大值、平均值
community_feas = ['area', 'mean_area', 'now_trade_interval',
                  'now_build_interval', 'totalFloor',
                  'tradeMeanPrice', 'tradeNewMeanPrice',
                  'totalTradeMoney', 'totalTradeArea', 'remainNewNum',
                  'uv_pv_ratio', 'pv', 'uv',
                  '室面积', '卫面积', '厅面积', '室数量', '厅数量', '卫数量', 'ID', '室卫厅数量'
                  ]

for fea in tqdm(community_feas):
    grouped_df = df.groupby('communityName').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
    grouped_df.columns = ['communityName_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)

    df = pd.merge(df, grouped_df, on='communityName', how='left')

for fea in tqdm(community_feas):
    an_df = df.groupby(['communityName', 'rentType']).agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
    an_df.columns = ['communityName_rentType_' + '_'.join(col).strip() for col in an_df.columns.values]
    an_df = an_df.reset_index()
    # print(grouped_df)

    df = pd.merge(df, an_df, on=['communityName', 'rentType'], how='left')

for fea in tqdm(community_feas):
    an_df = df.groupby(['communityName', 'houseDecoration']).agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
    an_df.columns = ['communityName_houseDecoration_' + '_'.join(col).strip() for col in an_df.columns.values]
    an_df = an_df.reset_index()
    # print(grouped_df)

    df = pd.merge(df, an_df, on=['communityName', 'houseDecoration'], how='left')

for fea in tqdm(community_feas):
    an_df = df.groupby(['communityName', 'houseType']).agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
    an_df.columns = ['communityName_houseType_' + '_'.join(col).strip() for col in an_df.columns.values]
    an_df = an_df.reset_index()
    # print(grouped_df)

    df = pd.merge(df, an_df, on=['communityName', 'houseType'], how='left')

# --------- 板块特征 -----------
for fea in tqdm(community_feas):
    grouped_df = df.groupby('plate').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
    grouped_df.columns = ['plate_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on='plate', how='left')

# # ----------- 地区特征 -------------
# region_trade_nums = dict(df['region'].value_counts())
# df['region_nums'] = df['region'].apply(lambda x: region_trade_nums[x])
#
# for fea in tqdm(community_feas):
#     grouped_df = df.groupby('region').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
#     grouped_df.columns = ['region_' + '_'.join(col).strip() for col in grouped_df.columns.values]
#     grouped_df = grouped_df.reset_index()
#
#     df = pd.merge(df, grouped_df, on='region', how='left')
#
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


# # ---------------- 建造年份 ---------------
# for fea in tqdm(community_feas):
#     grouped_df = df.groupby('buildYear').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
#     grouped_df.columns = ['buildYear_' + '_'.join(col).strip() for col in grouped_df.columns.values]
#     grouped_df = grouped_df.reset_index()
#     # print(grouped_df)
#     df = pd.merge(df, grouped_df, on='buildYear', how='left')


# 添加rank特征
cols = [col for col in (set(community_feas + numerical_feas))]
for col in cols:
    df[col + '_Rank'] = df[col].rank()

# 数量统计特征
need_num_feas = ['communityName', 'plate', 'tradeTime_season', 'region', 'tradeTime_month']
for mean_fea in need_num_feas:
    # 每个mean_fea的出现个数
    mean_fea_nums = dict(df[mean_fea].value_counts())
    df[mean_fea + '_nums'] = df[mean_fea].apply(lambda x: mean_fea_nums[x])

# 类比编码
categorical_feas = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'tradeTime_season']
# df = pd.get_dummies(df, columns=categorical_feas)
for col in categorical_feas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 生成数据
no_features = ['tradeTime', 'tradeMoney',
               'houseType', 'region', 'plate',
               'buildYear', 'communityName', 'city',
               'area_money', 'tradeTime_month'
               ]
no_features = no_features
features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(df_train)], df[len(df_train):]

print("训练集和测试集维度：", train.shape, test.shape)
df.head(100).to_csv('input/df.csv', index=False)
print(features)


def load_data():
    return train, test, no_features, features
