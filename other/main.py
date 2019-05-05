#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: main.py 
@time: 2019-05-06 02:25
@description:
"""

# -*- coding: utf-8 -*-
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import copy
import os
import gc
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import lightgbm as lgb
from datetime import datetime

scaler = StandardScaler()




# 显示所有列
# pd.set_option('display.max_columns', None)
#显示所有行
# pd.set_option('display.max_rows', None)
# #设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',100)

train = pd.read_csv('input/train_data.csv', parse_dates=['tradeTime'])
test = pd.read_csv('input/test_a.csv', parse_dates=['tradeTime'])

train = train[train['tradeMoney'] <= 100000].reset_index(drop=True)
train = train[train['tradeMoney'] >= 500].reset_index(drop=True)

train = train.query("10<=area<=200")
train = train.query("2<=totalFloor<=62")

community_cast = train['tradeMoney'].groupby(train['communityName']).mean().reset_index().rename(columns={'tradeMoney': 'community_cast'})
houseType_cast = train['tradeMoney'].groupby(train['houseType']).mean().reset_index().rename(columns={'tradeMoney': 'houseType_cast'})


label = train['tradeMoney']
del train['tradeMoney']
data = pd.concat((train, test))

# print(train.head())
# print(test.head())


rentType_le = LabelEncoder()
rentType_le.fit(['未知方式', '整租', '合租', '--'])


houseType_le = LabelEncoder()
houseType_le.fit(data['houseType'].unique())

houseFloor_le = LabelEncoder()
houseFloor_le.fit(data['houseFloor'].unique())

houseToward_le = LabelEncoder()
houseToward_le.fit(data['houseToward'].unique())

houseDecoration_le = LabelEncoder()
houseDecoration_le.fit(data['houseDecoration'].unique())

communityName_le = LabelEncoder()
communityName_le.fit(data['communityName'].unique())

buildYear_le = LabelEncoder()
buildYear_le.fit(data['buildYear'].unique())

region_le = LabelEncoder()
region_le.fit(data['region'].unique())

plate_le = LabelEncoder()
plate_le.fit(data['plate'].unique())


community_nums = data['ID'].groupby(data['communityName']).count().reset_index().rename(columns={'ID': 'community_nums'})
community_pv = data['pv'].groupby(data['communityName']).sum().reset_index().rename(columns={'ID': 'community_pv'})
# plate_pv = data['ID'].groupby(data['plate']).sum().reset_index().rename(columns={'ID': 'plate_nums'})

def get_fea(data, type='test'):
    data = pd.merge(data, community_nums, on='communityName', how='left')
    data = pd.merge(data, community_cast, on='communityName', how='left')
    # data = pd.merge(data, houseType_cast, on='houseType', how='left')
    # data = pd.merge(data, community_pv, on='communityName', how='left')
    # data = pd.merge(data, plate_pv, on='plate', how='left')

    data['卧室数量'] = data['houseType'].apply(lambda x: int(x[0]))
    data['厅数量'] = data['houseType'].apply(lambda x: int(x[2]))
    data['卫数量'] = data['houseType'].apply(lambda x: int(x[4]))

    data['all_nums'] = data['houseType'].apply(lambda x: int(x[0])+int(x[2])+int(x[4]))

    data = data.drop(['卧室数量'], axis=1)

    data['mean_area'] = data['area'] / data['all_nums']
    # data['look_all'] = (data['all_nums'] / data['lookNum']).replace(np.inf, -1)

    # time_end = datetime.strptime('2018-12-31', '%Y-%m-%d')
    # data['time_diff'] = data['tradeTime'].apply(lambda x: (time_end-+


    data['rentType'] = rentType_le.transform(data['rentType'])
    data['houseType'] = houseType_le.transform(data['houseType'])
    data['houseFloor'] = houseFloor_le.transform(data['houseFloor'])
    data['houseToward'] = houseToward_le.transform(data['houseToward'])
    data['houseDecoration'] = houseDecoration_le.transform(data['houseDecoration'])
    data['communityName'] = communityName_le.transform(data['communityName'])

    # data['buildYear'] = buildYear_le.transform(data['buildYear'])
    data['buildYear'] = data['buildYear'].apply(lambda x:np.nan if x == '暂无信息' else x)

    data['region'] = region_le.transform(data['region'])
    data['plate'] = plate_le.transform(data['plate'])

    data['tradeTime_month'] = data['tradeTime'].apply(lambda x: x.month)
    data['tradeTime_day'] = data['tradeTime'].apply(lambda x: x.day)

    del data['tradeTime']
    del data['city']

    # data['all_ss_nums'] = data['subwayStationNum'] + data['busStationNum'] + data['interSchoolNum'] + data['schoolNum']\
    #                       + data['privateSchoolNum'] + data['hospitalNum'] + data['drugStoreNum'] + data['gymNum']\
    #                       + data['bankNum'] + data['shopNum'] + data['parkNum'] + data['mallNum'] + data['superMarketNum']

    # onehot_cols = ['houseFloor']
    # data = pd.get_dummies(data, columns=onehot_cols)

    # if type == 'train':
    #     data = data.drop(['region_RG00015', 'plate_BK00032', 'plate_BK00058', 'plate_BK00001', 'all_nums_10', 'all_nums_11', 'all_nums_13', 'all_nums_14', 'all_nums_15', 'all_nums_16', 'all_nums_20'], axis=1)

    # data = data.drop(['rentType', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration', 'houseDecoration', 'communityName'
    #                   , 'buildYear', 'region', 'plate', 'tradeTime_month', 'tradeTime_day'], axis=1)

    return data




def get_result(train, test, label, my_model, need_sca=False, splits_nums=5):

    oof = np.zeros(train.shape[0])
    sub = test[['ID']]
    del train['ID']
    del test['ID']

    # print(train.head())

    if need_sca:
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    elif not need_sca:
        train = train.values
        test = test.values

    score_list = []

    label = np.array(label)

    test = xgb.DMatrix(test)


    k_fold = KFold(n_splits=splits_nums, shuffle=True, random_state=2019)
    for index, (train_index, test_index) in enumerate(k_fold.split(train)):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = my_model(X_train, y_train, X_test, y_test)

        X_test = xgb.DMatrix(X_test)
        vali_pre = model.predict(X_test)
        oof[test_index] = vali_pre
        score = r2_score(y_test, vali_pre)
        score_list.append(score)

        print(y_test)
        print(vali_pre)


        pred_result = model.predict(test)
        sub['tradeMoney'] = pred_result

        if index == 0:
            re_sub = copy.deepcopy(sub)
        else:
            re_sub['tradeMoney'] = re_sub['tradeMoney'] + sub['tradeMoney']

    re_sub['tradeMoney'] = re_sub['tradeMoney'] / splits_nums

    print('score list:', score_list)
    print(np.mean(score_list))
    return re_sub, oof

def xgb_model(X_train, y_train, X_test, y_test):
    xgb_params = {'eta': 0.01,
                  'max_depth': 6,
                  'subsample': 0.5,
                  'colsample_bytree': 0.5,
                  'alpha': 0.2,
                  'objective': 'reg:gamma',
                  'eval_metric': 'mae',
                  'silent': 1,
                  'nthread': -1
                  }

    trn_data = xgb.DMatrix(X_train, y_train)
    val_data = xgb.DMatrix(X_test, y_test)
    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]

    clf = xgb.train(dtrain=trn_data, num_boost_round=100000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)

    return clf


def xgb_model_re(X_train, y_train, X_test, y_test):

    model = xgb.XGBRegressor(colsample_bytree=0.3,
                             # objective= reg:gamma,
                             eval_metric='mae',
                             gamma=0.0,
                             learning_rate=0.01,
                             max_depth=4,
                             min_child_weight=1.5,
                             n_estimators=1668,
                             reg_alpha=1,
                             reg_lambda=0.6,
                             subsample=0.2,
                             seed=42,
                             silent=1)

    model = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
                      verbose=1)
    return model


def xgb_model_bs(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(max_depth=8,
                               learning_rate=0.1,
                               objective="reg:linear",
                               eval_metric='rmse',
                               n_estimators=3115,
                               colsample_bytree=0.6,
                               reg_alpha=3,
                               reg_lambda=2,
                               gamma=0.6,
                               subsample=0.7,
                               silent=1,
                               n_jobs=-1)
    model = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=100,
                      verbose=1)

    return model

train = get_fea(train, type='train')
test = get_fea(test)

print(list(train.columns))
# print(list(test.columns))

# print(label)

# xgb_model_re
sub, oof = get_result(train, test, label, xgb_model, need_sca=True, splits_nums=3)
# print(sub.sort_values('tradeMoney').reset_index(drop=True))
sub[['tradeMoney']].to_csv('output/xgb.csv', index=None, header=False)



# score list: [0.8834834505610115, 0.8248506813617509, 0.8539747905133632]  0.8541029741453752

# 0.8540411313811912

# score list: [0.8400237238337294, 0.897801296556761, 0.8906870036577093]
# 0.8761706746827332
