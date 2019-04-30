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


def keras_myscore(y_pred, y_true):
    return 1 - K.sum((y_pred - y_true) ** 2) / K.sum((y_pred - K.mean(y_true)) ** 2)


def xgb_score(y_pred, y_true):
    y_true = y_true.get_label()
    return 'my-error', 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2)
