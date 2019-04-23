#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 03_stack.py 
@time: 2019-04-23 23:50
@description:
"""
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from gen_feas import load_data

train, test, no_features, features = load_data()
features=['now_build_interval', 'area', 'tradeMeanPrice_area_multiply', 'totalFloor', 'tradeNewMeanPrice_area_multiply', 'tradeMeanPrice_area_div', 'totalTradeMoney_area_multiply', 'houseType_wei', 'totalTradeArea_area_multiply', 'now_trade_interval', 'totalTradeArea_area_div', 'houseType_shi', 'totalTradeMoney_area_div', 'tradeNewMeanPrice_area_div', 'houseToward', 'totalNewTradeMoney_area_div', 'houseFloor', 'totalNewTradeMoney_area_multiply', 'totalNewTradeArea_area_multiply', 'plate', 'houseType_ting', 'totalNewTradeArea_area_diff', 'totalNewTradeArea_area_div', 'houseDecoration', 'houseType', 'tradeMeanPrice_tradeNewMeanPrice_diff', 'pv_uv_div', 'tradeMeanPrice_area_diff', 'rentType', 'totalNewTradeArea_area_sum', 'totalTradeArea_tradeMeanPrice_diff', 'tradeMeanPrice_totalTradeMoney_multiply', 'totalTradeArea_totalTradeMoney_div', 'tradeSecNum_newWorkers_sum', 'tradeSecNum_totalWorkers_div', 'tradeNewMeanPrice_area_diff', 'tradeMeanPrice_tradeNewMeanPrice_sum', 'busStationNum_gymNum_div', 'saleSecHouseNum_privateSchoolNum_diff', 'remainNewNum_totalWorkers_multiply', 'busStationNum_tradeSecNum_diff', 'gymNum_bankNum_div', 'pv', 'totalTradeArea_tradeMeanPrice_sum', 'drugStoreNum_gymNum_div', 'shopNum_superMarketNum_div', 'busStationNum_parkNum_div', 'totalTradeMoney', 'tradeNewMeanPrice_totalNewTradeMoney_div', 'tradeSecNum_totalWorkers_multiply']

lr = LinearRegression()
model_xgb = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=15,
    reg_lambda=0.01,
    reg_alpha=0.01,
    colsample_bytree=0.85,
    min_child_samples=24,
    num_leaves=60,
)

model_lgb = lgb.LGBMRegressor(
    # 下面注释的是另一种参数选择
    # boosting_type='dart',n_estimators=10000,num_leaves=60,max_depth=10,
    n_estimators=1000,
    reg_lambda=0.01,
    max_depth=16,
    min_child_weight=0.001,
    reg_alpha=0.01,
    colsample_bytree=0.85,
    min_child_samples=24,
    max_bin=500,
    num_leaves=45,)

X = train[features].values
y = train['tradeMoney'].values
print(y)
test_data = test[features].values

# 融合两个模型
stregr = StackingRegressor(regressors=[model_lgb, model_xgb], meta_regressor=lr,verbose=3)


# 训练stacking分类器
stregr.fit(X, y)
stregr_pred = stregr.predict(test_data)
result = pd.DataFrame()
result['p'] = stregr_pred
result.to_csv('stacking_new.csv', header=False,index=False)
