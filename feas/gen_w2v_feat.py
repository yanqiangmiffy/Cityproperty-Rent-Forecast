#!/usr/bin/env python
# encoding: utf-8
"""
@author: quincyqiang
@software: PyCharm
@file: gen_w2v_feat.py
@time: 2019-05-17 09:56
@description: 生成Word2Vec特征
"""

import pandas as pd
import warnings
from gensim.models import Word2Vec
import multiprocessing

warnings.filterwarnings('ignore')


def w2v_feat(data_frame, feat, mode):
    for i in feat:
        if data_frame[i].dtype != 'object':
            data_frame[i] = data_frame[i].astype(str)
    data_frame.fillna('nan', inplace=True)

    print(f'Start {mode} word2vec ...')
    model = Word2Vec(data_frame[feat].values.tolist(), size=L, window=2, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=10)
    stat_list = ['min', 'max', 'mean', 'std']
    new_all = pd.DataFrame()
    for m, t in enumerate(feat):
        print(f'Start gen feat of {t} ...')
        tmp = []
        for i in data_frame[t].unique():
            tmp_v = [i]
            tmp_v.extend(model[i])
            tmp.append(tmp_v)
        tmp_df = pd.DataFrame(tmp)
        w2c_list = [f'w2c_{t}_{n}' for n in range(L)]
        tmp_df.columns = [t] + w2c_list
        tmp_df = data_frame[['ID', t]].merge(tmp_df, on=t)
        tmp_df = tmp_df.drop_duplicates().groupby('ID').agg(stat_list).reset_index()
        tmp_df.columns = ['ID'] + [f'{p}_{q}' for p in w2c_list for q in stat_list]
        if m == 0:
            new_all = pd.concat([new_all, tmp_df], axis=1)
        else:
            new_all = pd.merge(new_all, tmp_df, how='left', on='ID')
    return new_all


if __name__ == '__main__':
    L = 10
    df_train = pd.read_csv('../input/train_data.csv')
    df_test = pd.read_csv('../input/test_a.csv')
    # ------------------ 过滤数据 begin ----------------
    print("根据tradeMoney过滤数据:", len(df_train))
    df_train = df_train.query("500<=tradeMoney<25000")  # 线下 lgb_0.876612870005764
    print("filter tradeMoney after:", len(df_train))

    categorical_feas = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration']
    new_all_train = w2v_feat(df_train, categorical_feas, 'train')
    new_all_test = w2v_feat(df_test, categorical_feas, 'test')
    train = pd.merge(df_train, new_all_train, on='ID', how='left')
    valid = pd.merge(df_test, new_all_test, on='ID', how='left')
    print(f'Gen train shape: {train.shape}, test shape: {valid.shape}')

    drop_train = train.T.drop_duplicates().T
    drop_valid = valid.T.drop_duplicates().T

    features = [i for i in drop_train.columns if i in drop_valid.columns]
    print('features num: ', len(features) - 1)
    train[features + ['tradeMoney']].to_csv('../input/train_w2v.csv', index=False)
    valid[features].to_csv('../input/test_w2v.csv', index=False)
