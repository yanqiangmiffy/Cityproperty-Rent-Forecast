#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: analysis_result.py 
@time: 2019-04-26 16:44
@description: 结果分析 从交叉验证上面分析误差
"""
import pandas as pd
pd.set_option('display.max_columns',100)
true_pred=pd.read_csv('output/full_df.csv')
true_pred['error']=abs(true_pred['tradeMoney']-true_pred['pred_tradeMoney'])
# print(true_pred['error'])
print(true_pred[true_pred['error']>=10000])