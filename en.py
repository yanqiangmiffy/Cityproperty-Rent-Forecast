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

lgb=pd.read_csv('output/lgb_0.9032697811798844.csv',header=None)
xgb=pd.read_csv('output/xgb.csv',header=None)

# res=lgb*0.6+xgb*0.4 # 0.872517
res=lgb*0.7+xgb*0.3   # 0.877785
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('output/en73.csv',index=None,header=None)
