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
df=pd.read_csv('output/full_df.csv')
df=df[['area','communityName','tradeMoney']]
df_test=pd.read_csv('input/test_a.csv')
df_pred=pd.read_csv('output/lgb_0.92124176467958.csv',header=None)
df_test['pred']=df_pred.values

commnity_mean=dict()
for index,group in df.groupby(by='communityName'):
    commnity_mean[index]=group['tradeMoney'].mean()

for index,group in df_test.groupby(by='communityName'):
    if index not in commnity_mean:
        commnity_mean[index]=group['pred'].mean()
df_test['mean']=df_test['communityName'].apply(lambda x:commnity_mean[x])

df_test['sub']=df_test['pred']*0.8+df_test['mean']*0.2
df_test['sub'].to_csv('result_correction.csv',index=None,header=None)