#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: 05_sklearn_nn.py
@time: 2019-05-14 10:16
@description:
"""
from sklearn.neural_network import MLPRegressor
from gen_feas import load_data
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time
from utils import *
import pandas as pd

# ----------加载数据------------
train, test, no_features, features = load_data()

#
X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values

res_list = []  # 结果
scores_list = []

print("start：********************************")
start = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=2019)
for k, (train_index, test_index) in enumerate(kf.split(X, y)):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]

    est = MLPRegressor(hidden_layer_sizes=(100,), activation="relu",
                       solver='adam', alpha=0.0001,
                       batch_size='auto', learning_rate="constant",
                       learning_rate_init=0.001,
                       power_t=0.5, max_iter=200, shuffle=True, )
    est.fit(X, y)
    # 验证集测试
    print('................Start predict .........................')
    valid_pred = est.predict(x_valid)
    score = r2_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = est.predict(test_data)
    res_list.append(pred)

print('......................validate result mean :', np.mean(scores_list))
end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 线下平均分数
mean_score = np.mean(scores_list)
print("lgb mean score:", mean_score)
filepath = 'output/lgb_' + str(mean_score) + '.csv'  #

# 提交结果 5折平均
print("lgb 提交结果...")
res = np.array(res_list)
r = res.mean(axis=0)
result = pd.DataFrame()
result['p'] = r
result.to_csv(filepath, header=False, index=False, sep=",")
