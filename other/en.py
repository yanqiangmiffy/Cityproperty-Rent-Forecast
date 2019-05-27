#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: en.py 
@time: 2019-04-30 22:46
@description:
"""
import pandas as pd

lgb = pd.read_csv('../output/lgb_0.9382180135973881.csv', header=None)
ebm = pd.read_csv('../output/lgb_0.9288663947077677.csv', header=None)
xgb = pd.read_csv('../output/sub_xgb_revise_lstm_qiang_yann_fei3.csv', header=None)

# res=lgb*0.6+xgb*0.4 # 0.872517
res = lgb * 0.2 + xgb * 0.6 +ebm*0.2# 0.877785
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('../output/en73.csv', index=None, header=None)
