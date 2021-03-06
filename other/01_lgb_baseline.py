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
# from feas.gen_feas5 import load_data
from gen_feas import load_data
from utils import my_score

train, test, no_features, features = load_data()
print(train.head())
print(len(features))
# features=['now_build_interval', 'area', 'tradeMeanPrice_area_multiply', 'totalFloor', 'tradeNewMeanPrice_area_multiply', 'tradeMeanPrice_area_div', 'totalTradeMoney_area_multiply', 'houseType_wei', 'totalTradeArea_area_multiply', 'now_trade_interval', 'totalTradeArea_area_div', 'houseType_shi', 'totalTradeMoney_area_div', 'tradeNewMeanPrice_area_div', 'houseToward', 'totalNewTradeMoney_area_div', 'houseFloor', 'totalNewTradeMoney_area_multiply', 'totalNewTradeArea_area_multiply', 'plate', 'houseType_ting', 'totalNewTradeArea_area_diff', 'totalNewTradeArea_area_div', 'houseDecoration', 'houseType', 'tradeMeanPrice_tradeNewMeanPrice_diff', 'pv_uv_div', 'tradeMeanPrice_area_diff', 'rentType', 'totalNewTradeArea_area_sum', 'totalTradeArea_tradeMeanPrice_diff', 'tradeMeanPrice_totalTradeMoney_multiply', 'totalTradeArea_totalTradeMoney_div', 'tradeSecNum_newWorkers_sum', 'tradeSecNum_totalWorkers_div', 'tradeNewMeanPrice_area_diff', 'tradeMeanPrice_tradeNewMeanPrice_sum', 'busStationNum_gymNum_div', 'saleSecHouseNum_privateSchoolNum_diff', 'remainNewNum_totalWorkers_multiply', 'busStationNum_tradeSecNum_diff', 'gymNum_bankNum_div', 'pv', 'totalTradeArea_tradeMeanPrice_sum', 'drugStoreNum_gymNum_div', 'shopNum_superMarketNum_div', 'busStationNum_parkNum_div', 'totalTradeMoney', 'tradeNewMeanPrice_totalNewTradeMoney_div', 'tradeSecNum_totalWorkers_multiply']
# features=['now_build_interval', 'totalFloor', 'area', 'tradeMeanPrice_area_multiply', 'now_trade_interval', 'tradeNewMeanPrice_area_multiply', 'tradeMeanPrice_area_div', 'totalTradeMoney_area_multiply', 'totalTradeArea_area_multiply', 'totalTradeMoney_area_div', 'totalTradeArea_area_div', 'tradeNewMeanPrice_area_div', 'houseType_shi', 'houseFloor', 'houseDecoration', 'houseToward', 'totalNewTradeMoney_area_multiply', 'totalNewTradeArea_area_multiply', 'houseType_wei', 'totalNewTradeMoney_area_div', 'totalNewTradeArea_area_div', 'houseType_ting', 'pv_uv_div', 'plate', 'tradeNewMeanPrice_area_diff', 'houseType', 'totalNewTradeArea_area_diff', 'rentType', 'uv', 'tradeMeanPrice_tradeNewMeanPrice_div', 'tradeMeanPrice_tradeNewMeanPrice_diff', 'pv', 'uv_lookNum_div', 'tradeNewMeanPrice_area_sum', 'tradeMeanPrice_totalNewTradeArea_sum', 'tradeNewMeanPrice_totalNewTradeArea_div', 'pv_lookNum_multiply', 'tradeNewMeanPrice', 'tradeMeanPrice', 'totalTradeArea_tradeNewMeanPrice_diff', 'tradeMeanPrice_tradeNewMeanPrice_sum', 'totalTradeArea_tradeNewMeanPrice_sum', 'pv_lookNum_div', 'busStationNum_hospitalNum_div', 'privateSchoolNum_tradeSecNum_div', 'tradeSecNum_totalWorkers_div', 'totalTradeArea_tradeMeanPrice_diff', 'totalTradeArea_tradeMeanPrice_sum', 'tradeNewMeanPrice_totalNewTradeArea_diff', 'saleSecHouseNum_totalWorkers_multiply']

X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values

# 训练集的预测结果
valid_id=[]
valid_index=[]
valid_list=[]
valid_pred_list=[]

res_list = []
scores_list = []

print("start：********************************")
start = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=2019)
for train_index, test_index in kf.split(X, y):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]
    valid_id.extend(list(train.ID[test_index].values))
    clf = LGBMRegressor(
        # boosting_type='gbdt',
        #                     num_leaves=64,
        #                     # max_depth=5,
        #                     learning_rate=0.02,
        #                     n_estimators=2000
        #                     boosting_type='dart',n_estimators=15000,num_leaves=100,max_depth=12
        objective='regression', num_leaves=90,
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
    valid_index.extend(list(test_index))
    valid_list.extend(list(y_valid))
    valid_pred_list.extend(list(valid_pred))
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
plt.savefig('other/lgbm_importances-01.png')
plt.show()


raw_df=pd.read_csv('input/train_data.csv')

valid_df=pd.DataFrame()
valid_df['ID']=valid_id
valid_df['pred_tradeMoney']=valid_pred_list

full_df=pd.merge(raw_df,valid_df,on="ID")
full_df['error']=full_df['tradeMoney']-full_df['pred_tradeMoney']
full_df.to_csv('output/full_df.csv',index=None)
