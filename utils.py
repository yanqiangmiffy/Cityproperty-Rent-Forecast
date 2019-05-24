#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: utils.py 
@time: 2019-04-23 15:20
@description:
"""
import numpy as np
import keras.backend as K


# 定义评价函数
def my_score(y_pred, y_true):
    # y_true = y_true.get_label()
    return 'my_score', 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2), True


# 定义评价函数
def lgb_score(y_pred, y_true):
    y_true = y_true.get_label()
    return 'lgb_score', 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2), True


def xgb_score(y_pred, y_true):
    y_true = y_true.get_label()
    return 'xgb_score', 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2)


def keras_socre(y_true, y_pred):
    # y_true = y_true.get_label()
    print(y_true)
    print(y_pred)
    return 'keras_score', 1 - K.sum((y_pred - y_true) ** 2) / K.sum((y_pred - K.mean(y_true)) ** 2)


numerical_feas = ['area', 'totalFloor', 'saleSecHouseNum', 'subwayStationNum',
                  'busStationNum', 'interSchoolNum', 'schoolNum', 'privateSchoolNum', 'hospitalNum',
                  'drugStoreNum', 'gymNum', 'bankNum', 'shopNum', 'parkNum', 'mallNum', 'superMarketNum',
                  'totalTradeMoney', 'totalTradeArea', 'tradeMeanPrice', 'tradeSecNum', 'totalNewTradeMoney',
                  'totalNewTradeArea', 'tradeNewMeanPrice', 'tradeNewNum', 'remainNewNum', 'supplyNewNum',
                  'supplyLandNum', 'supplyLandArea', 'tradeLandNum', 'tradeLandArea', 'landTotalPrice',
                  'landMeanPrice', 'totalWorkers', 'newWorkers', 'residentPopulation', 'pv', 'uv', 'lookNum']
