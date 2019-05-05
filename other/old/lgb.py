#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: lgb.py 
@time: 2019-05-05 20:14
@description:
"""
#!/usr/bin/env python
# -*- coding:utf-8 _*-

import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
train=pd.read_csv('input/train_data.csv')
test=pd.read_csv('input/test_a.csv')
print(len(train),len(test))
df=pd.concat([train,test],keys="ID",axis=0,sort=True)

no_features=['ID','tradeTime','tradeMoney','buildYear','region','plate','communityName']
categorical_feas=['rentType','houseType','houseFloor','houseToward','houseDecoration','city']
df=pd.get_dummies(df,columns=categorical_feas)
train,test=df[:len(train)],df[len(train):]
# train=pd.get_dummies(train,columns=categorical_feas)
# test=pd.get_dummies(test,columns=categorical_feas)
features=[fea for fea in train.columns if fea not in no_features]
print(features)
train.head().to_csv('demo.csv')
# 8.得到输入X ，输出y
train_id = train['ID'].values
y = train['tradeMoney'].values.astype("float32")
print(y)
X = train[features].values
print("X shape:",X.shape)
print("y shape:",y.shape)

test_id = test['ID'].values
test_data = test[features].values
print("test shape",test_data.shape)

# 9.开始训练
# 采取分层采样
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

print("start：********************************")
start = time.time()

N = 5
skf = KFold(n_splits=N,shuffle=True,random_state=2018)

auc_cv = []
pred_cv = []
for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], \
                                       y[train_in], y[test_in]

    # 数据结构
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        # 'metric': {'mse','l2', 'l1'},
        'metric': {'rmse'},
        'num_leaves': 167,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('................Start training {} fold..........................'.format(k+1))
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_eval,
                    # early_stopping_rounds=100,
                    verbose_eval=100,feature_name=features)
    lgb.plot_importance(gbm,max_num_features=20)
    plt.show()
    print('................Start predict .........................')
    # 预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # 评估
    tmp_auc = mean_squared_error(y_test, y_pred)
    auc_cv.append(tmp_auc)
    print("valid auc:", tmp_auc)
    # test
    pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    pred_cv.append(pred)

    # K交叉验证的平均分数
print('the cv information:')
print(auc_cv)
print('cv mean score', np.mean(auc_cv))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 10.5折交叉验证结果均值融合，保存文件
mean_auc = np.mean(auc_cv)
print("mean auc:", mean_auc)
filepath = 'output/lgb_' + str(mean_auc) + '.csv'  # 线下平均分数

# 转为array
res = np.array(pred_cv)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = pd.DataFrame()
# result['ID'] = test_id
result['p'] = r
result.to_csv(filepath, header=False,index=False, sep=",")
