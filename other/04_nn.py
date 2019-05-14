#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: 04_nn.py
@time: 2019-05-13 22:49
@description:
"""
import pandas as pd
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam, RMSprop
from gen_feas import load_data
from sklearn.model_selection import KFold
import time
from utils import *

# ----------加载数据------------
train, test, no_features, features = load_data()

#
X = train[features].values
y = train['tradeMoney'].values
test_data = test[features].values


def create_mlp(dim, ):
    # define our MLP network
    model = Sequential()
    model.add(Dense(64, input_dim=dim, activation="relu"))
    # model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model


# ----------lgb------------

res_list = []  # 结果
scores_list = []

print("start：********************************")
start = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=2019)
for k, (train_index, test_index) in enumerate(kf.split(X, y)):
    x_train, y_train = X[train_index], y[train_index]
    x_valid, y_valid = X[test_index], y[test_index]

    mlp = create_mlp(X.shape[1])
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    mlp.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=['mae'])
    mlp.summary()
    # 训练模型
    print("[INFO] training model...")
    mlp.fit(X, y, epochs=10, batch_size=64)

    # 验证集测试
    print('................Start predict .........................')
    valid_pred = mlp.predict(x_valid)
    score = my_score(y_valid, valid_pred)
    print("------------ r2_score:", score)
    scores_list.append(score)

    # 测试集预测
    pred = mlp.predict(test_data)
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
