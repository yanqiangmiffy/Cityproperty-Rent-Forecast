#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 01_lgb_baseline.py
@time: 2019-04-23 15:18
@description: lgb 模型
"""
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from gen_feas import load_data
from utils import my_score

train, test, no_features, features = load_data()
print(train.head())
print(len(features))
# features=['now_build_interval', 'area', 'tradeMeanPrice_area_multiply', 'totalFloor', 'now_trade_interval', 'tradeMeanPrice_area_div', 'tradeNewMeanPrice_area_multiply', 'totalTradeArea_area_multiply', 'totalTradeArea_area_div', 'totalTradeMoney_area_multiply', 'totalTradeMoney_area_div', 'tradeNewMeanPrice_area_div', 'houseFloor', 'houseType_shi', 'houseDecoration', 'totalNewTradeArea_area_div', 'totalNewTradeMoney_area_div', 'totalNewTradeArea_area_multiply', 'pv_uv_div', 'totalNewTradeMoney_area_multiply', 'houseToward', 'houseType_ting', 'pv', 'uv', 'plate', 'tradeMeanPrice', 'tradeNewMeanPrice_area_diff', 'tradeMeanPrice_tradeNewMeanPrice_div', 'houseType_wei', 'houseType', 'tradeMeanPrice_totalNewTradeArea_sum', 'uv_lookNum_div', 'tradeMeanPrice_tradeNewMeanPrice_diff', 'totalNewTradeArea_area_diff', 'tradeNewMeanPrice_area_sum', 'totalTradeArea_tradeMeanPrice_sum', 'rentType', 'privateSchoolNum_tradeSecNum_div', 'totalTradeArea_tradeNewMeanPrice_sum', 'totalTradeArea_tradeNewMeanPrice_diff', 'tradeMeanPrice_tradeNewMeanPrice_sum', 'subwayStationNum_tradeSecNum_div', 'totalNewTradeArea_area_sum', 'totalTradeArea_tradeMeanPrice_diff', 'tradeSecNum_newWorkers_sum', 'tradeSecNum_totalWorkers_div', 'totalNewTradeArea_totalNewTradeMoney_div', 'tradeMeanPrice_totalNewTradeArea_diff', 'tradeMeanPrice_totalTradeMoney_multiply', 'tradeNewMeanPrice']
# features=['now_build_interval', 'area', 'tradeMeanPrice_area_multiply', 'totalFloor', 'tradeNewMeanPrice_area_multiply', 'tradeMeanPrice_area_div', 'totalTradeMoney_area_multiply', 'houseType_wei', 'totalTradeArea_area_multiply', 'now_trade_interval', 'totalTradeArea_area_div', 'houseType_shi', 'totalTradeMoney_area_div', 'tradeNewMeanPrice_area_div', 'houseToward', 'totalNewTradeMoney_area_div', 'houseFloor', 'totalNewTradeMoney_area_multiply', 'totalNewTradeArea_area_multiply', 'plate', 'houseType_ting', 'totalNewTradeArea_area_diff', 'totalNewTradeArea_area_div', 'houseDecoration', 'houseType', 'tradeMeanPrice_tradeNewMeanPrice_diff', 'pv_uv_div', 'tradeMeanPrice_area_diff', 'rentType', 'totalNewTradeArea_area_sum', 'totalTradeArea_tradeMeanPrice_diff', 'tradeMeanPrice_totalTradeMoney_multiply', 'totalTradeArea_totalTradeMoney_div', 'tradeSecNum_newWorkers_sum', 'tradeSecNum_totalWorkers_div', 'tradeNewMeanPrice_area_diff', 'tradeMeanPrice_tradeNewMeanPrice_sum', 'busStationNum_gymNum_div', 'saleSecHouseNum_privateSchoolNum_diff', 'remainNewNum_totalWorkers_multiply', 'busStationNum_tradeSecNum_diff', 'gymNum_bankNum_div', 'pv', 'totalTradeArea_tradeMeanPrice_sum', 'drugStoreNum_gymNum_div', 'shopNum_superMarketNum_div', 'busStationNum_parkNum_div', 'totalTradeMoney', 'tradeNewMeanPrice_totalNewTradeMoney_div', 'tradeSecNum_totalWorkers_multiply']
# features=['saleSecHouseNum_tradeMoney_diff', 'tradeLandNum_tradeMoney_sum', 'supplyLandNum_tradeMoney_sum', 'interSchoolNum_tradeMoney_sum', 'saleSecHouseNum_tradeMoney_sum', 'supplyLandNum_tradeMoney_diff', 'lookNum_tradeMoney_sum', 'tradeLandNum_tradeMoney_diff', 'tradeMoney_area_diff', 'tradeMoney_houseType_wei_div', 'lookNum_tradeMoney_diff', 'interSchoolNum_tradeMoney_diff', 'tradeMoney_houseType_shi_diff', 'tradeMoney_houseType_ting_sum', 'tradeMoney_area_sum', 'subwayStationNum_tradeMoney_sum', 'tradeMoney_houseType_ting_multiply', 'newWorkers_tradeMoney_sum', 'tradeMoney_houseType_wei_sum', 'tradeMoney_houseType_ting_diff', 'tradeMoney_houseType_shi_div', 'tradeMoney_houseType_wei_multiply', 'tradeMoney_houseType_ting_div', 'hospitalNum_tradeMoney_diff', 'tradeMoney_houseType_wei_diff', 'tradeMoney_now_build_interval_sum', 'newWorkers_tradeMoney_diff', 'parkNum_tradeMoney_sum', 'gymNum_tradeMoney_diff', 'landMeanPrice_tradeMoney_sum', 'supplyNewNum_tradeMoney_diff', 'supplyNewNum_tradeMoney_sum', 'tradeMoney_tradeTime_season_sum', 'tradeMoney_now_trade_interval_diff', 'mallNum_tradeMoney_sum', 'tradeMoney_totalFloor_sum', 'bankNum_tradeMoney_sum', 'now_trade_interval_area_diff', 'now_trade_interval_tradeTime_month_multiply', 'tradeMoney_houseType_shi_sum', 'privateSchoolNum_tradeMoney_sum', 'tradeMoney_now_build_interval_diff', 'mallNum_tradeMoney_diff', 'hospitalNum_tradeMoney_sum', 'supplyLandArea_tradeMoney_sum', 'drugStoreNum_tradeMoney_sum', 'bankNum_tradeMoney_diff', 'tradeMoney_houseType_shi_multiply', 'schoolNum_tradeMoney_diff', 'privateSchoolNum_tradeMoney_diff']

X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values

res_list = []
scores_list = []

print("start：********************************")
start = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=2019)
for train_index, test_index in kf.split(X, y):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]

    clf = LGBMRegressor(
        # boosting_type='gbdt',
        #                     num_leaves=64,
        #                     # max_depth=5,
        #                     learning_rate=0.02,
        #                     n_estimators=2000
        #                     boosting_type='dart',n_estimators=15000,num_leaves=100,max_depth=12
        objective='regression', num_leaves=900,
        learning_rate=0.02, n_estimators=2000, bagging_fraction=0.7,
        feature_fraction=0.6, reg_alpha=0.3, reg_lambda=0.3,
        min_data_in_leaf=18, min_sum_hessian_in_leaf=0.001, n_jobs=-1)
    clf.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=my_score,
            early_stopping_rounds=100,
            verbose=True)
    # 验证集测试
    valid_pred = clf.predict(x_valid)
    score = r2_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = clf.predict(test_data)
    res_list.append(pred)

print('......................validate result mean :', np.mean(scores_list))
end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 11.5折结果均值融合，并保存文件
mean_auc = np.mean(scores_list)
print("mean auc:", mean_auc)
filepath = 'output/lgb_' + str(mean_auc) + '.csv'  # 线下平均分数

# 转为array
res = np.array(res_list)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = pd.DataFrame()
# result['ID'] = test_id
result['p'] = r
result.to_csv(filepath, header=False, index=False, sep=",")

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, features)), columns=['Value', 'Feature'])
print(list(feature_imp.sort_values(by="Value", ascending=False)['Feature'][:50]))
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:50])
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances-01.png')
plt.show()
