# -*- coding:utf-8 _*-
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import copy
import lightgbm as lgb
from tqdm import tqdm


# pd.set_option('display.max_rows', None)

scaler = StandardScaler()


df_train = pd.read_csv('input/train_data.csv', parse_dates=['tradeTime'])
df_test = pd.read_csv('input/test_a.csv', parse_dates=['tradeTime'])



unique_comname = df_test['communityName'].unique()

unique_house = df_test['houseType'].unique()

def get_train(df_train):

    df_train['area_money'] = df_train['tradeMoney'] / df_train['area']

    df_train = df_train.query("15<=area_money<300")

    df_train = df_train.query("500<=tradeMoney<20000")  # 线下 lgb_0.876612870005764

    df_train = df_train.query("15<=area<=150")  # 线下 lgb_0.8830538988139025 线上0.867

    df_train['area_money']=df_train['tradeMoney']/df_train['area']

    df_train = df_train.query("15<=area_money<300")  # 线下 lgb_0.9003567192921244.csv 线上0.867649

    df_train = df_train.query("2<=totalFloor<=53")

    df_train = df_train[df_train['region'] != 'RG00015']

    df_train = df_train[df_train['plate'] != 'BK00032']

    return df_train

def get_fea(df):
    df['landMeanPrice'] = df['landMeanPrice'].apply(lambda x: int(x))
    df['tradeTime_month'] = df['tradeTime'].apply(lambda x: x.month)
    df['tradeTime_day'] = df['tradeTime'].apply(lambda x: x.day)

    community_nums = df['ID'].groupby(df['communityName']).count().reset_index().rename(
        columns={'ID': 'community_nums'})
    df = pd.merge(df, community_nums, on='communityName', how='left')

    plate_nums = df['ID'].groupby(df['plate']).count().reset_index().rename(
        columns={'ID': 'plate_nums'})
    df = pd.merge(df, plate_nums, on='plate', how='left')


    df['all_nums'] = df['houseType'].apply(lambda x: int(x[0]) + int(x[2]) + int(x[4]))

    df['mean_area'] = df['area'] / df['all_nums']


    # 数据预处理
    df['rentType'] = df['rentType'].replace('--', '未知方式')


    df = pd.merge(df, area_money_plate, on='plate', how='left')
    df = pd.merge(df, area_money_plate_std, on='plate', how='left')
    df = pd.merge(df, area_money_plate_max, on='plate', how='left')
    df = pd.merge(df, area_money_plate_min, on='plate', how='left')


    communityName_le = LabelEncoder()
    communityName_le.fit(df['communityName'].unique())
    df['communityName'] = communityName_le.transform(df['communityName'])

    def split_type(x):
        """
        分割房屋类型
        :param x:
        :return:
        """
        assert len(x) == 6, "x的长度必须为6"
        return int(x[0]), int(x[2]), int(x[4])


    df['houseType_shi'], df['houseType_ting'], df['houseType_wei'] = zip(*df['houseType'].apply(lambda x: split_type(x)))

    df['w_area'] = df['area']/df['houseType_shi']

    # 交易至今的天数
    df['now_trade_interval'] = df['tradeTime'].apply(lambda x: (datetime.strptime('2019-05-12', '%Y-%m-%d')-x).days)


    df['buildYear'] = df['buildYear'].replace('暂无信息', 0)
    df['buildYear'] = df['buildYear'].astype(int)


    buildyear_median = dict()
    buildyear_mean = dict()
    buildyear_mode = dict()
    df_buildYear = df[df['buildYear'] != 0].reset_index(drop=True)
    for index, group in df_buildYear[['plate', 'buildYear']].groupby("plate"):
        buildyear_median[index] = int(group['buildYear'].median())
        buildyear_mean[index] = int(group['buildYear'].mean())
        buildyear_mode[index] = int(group['buildYear'].value_counts().index[0])

    def replace_zero(row):
        if row.buildYear == 0:
            return buildyear_mean[row.plate]
        else:
            return row.buildYear

    # 使用相同板块内的中位数年份、平均数年份、众数年份
    df['buildYear'] = df.apply(lambda row: replace_zero(row), axis=1)
    df['now_build_interval'] = 2019 - df['buildYear']


    tradeMeanPrice_mean = dict()
    df_tradeMeanPrice = df[df['tradeMeanPrice'] != 0].reset_index(drop=True)
    for index, group in df_tradeMeanPrice[['region', 'tradeMeanPrice']].groupby("region"):
        tradeMeanPrice_mean[index] = int(group['tradeMeanPrice'].mean())

    def replace_zero(row):
        if row.tradeMeanPrice == 0:
            return tradeMeanPrice_mean[row.region]
        else:
            return row.tradeMeanPrice

    df['tradeMeanPrice'] = df.apply(lambda row: replace_zero(row), axis=1)

    # 缺失值处理
    df['pv'] = df['pv'].fillna(value=int(df['pv'].median()))
    df['uv'] = df['uv'].fillna(value=int(df['uv'].median()))


    rentType_le = LabelEncoder()
    rentType_le.fit(['未知方式', '整租', '合租', '--'])

    houseType_le = LabelEncoder()
    houseType_le.fit(df['houseType'].unique())

    houseFloor_le = LabelEncoder()
    houseFloor_le.fit(df['houseFloor'].unique())

    houseToward_le = LabelEncoder()
    houseToward_le.fit(df['houseToward'].unique())

    houseDecoration_le = LabelEncoder()
    houseDecoration_le.fit(df['houseDecoration'].unique())

    communityName_le = LabelEncoder()
    communityName_le.fit(df['communityName'].unique())

    region_le = LabelEncoder()
    region_le.fit(df['region'].unique())

    plate_le = LabelEncoder()
    plate_le.fit(df['plate'].unique())

    df['rentType'] = rentType_le.transform(df['rentType'])
    df['houseType'] = houseType_le.transform(df['houseType'])
    df['houseFloor'] = houseFloor_le.transform(df['houseFloor'])
    df['houseToward'] = houseToward_le.transform(df['houseToward'])
    df['houseDecoration'] = houseDecoration_le.transform(df['houseDecoration'])
    df['communityName'] = communityName_le.transform(df['communityName'])

    df['region'] = region_le.transform(df['region'])
    df['plate'] = plate_le.transform(df['plate'])


    categorical_feas = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'region']
    df = pd.get_dummies(df, columns=categorical_feas)


    # 其他特征
    df['stationNum'] = df['subwayStationNum'] + df['busStationNum']
    df['schoolNum'] = df['interSchoolNum'] + df['schoolNum'] + df['privateSchoolNum']
    df['medicalNum'] = df['hospitalNum'] + df['drugStoreNum']
    df['lifeHouseNum'] = df['gymNum'] + df['bankNum'] + df['shopNum'] + df['parkNum'] + df['mallNum'] + df['superMarketNum']

    # 重要特征
    df['area_floor_ratio'] = df['area'] / (df['totalFloor'] + 1)
    df['uv_pv_ratio'] = df['uv'] / (df['pv'] + 1)
    df['uv_pv_sum'] = df['uv'] + df['pv']


    community_feas = ['area', 'mean_area', 'now_trade_interval',
                      'now_build_interval', 'totalFloor',
                      'tradeMeanPrice', 'tradeNewMeanPrice',
                      'totalTradeMoney', 'totalTradeArea', 'remainNewNum',
                      'uv_pv_ratio', 'pv', 'uv', 'tradeTime_month'
                      ]

    for fea in tqdm(community_feas):
        grouped_df = df.groupby('communityName').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
        grouped_df.columns = ['communityName_' + '_'.join(col).strip() for col in grouped_df.columns.values]
        grouped_df = grouped_df.reset_index()
        df = pd.merge(df, grouped_df, on='communityName', how='left')

    for fea in tqdm(community_feas):
        grouped_df = df.groupby('plate').agg({fea: ['min', 'max', 'mean', 'sum', 'median']})
        grouped_df.columns = ['plate_' + '_'.join(col).strip() for col in grouped_df.columns.values]
        grouped_df = grouped_df.reset_index()
        df = pd.merge(df, grouped_df, on='plate', how='left')

    no_features = ['ID', 'tradeTime', 'tradeMoney', 'newWorkers',
                   'buildYear', 'city', 'area_money', 'busStationNum', 'stationNum', 'lookNum', 'saleSecHouseNum', 'newWorkers'
                   ]

    no_features = no_features

    features = [fea for fea in df.columns if fea not in no_features]
    train, test = df[:len(df_train)], df[len(df_train):]

    return train, test, no_features, features


def get_result(train, test, label, my_model, need_sca=True, splits_nums=5):

    oof = np.zeros(train.shape[0])

    if need_sca:
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
    elif not need_sca:
        train = train.values
        test = test.values

    score_list = []

    label = np.array(label)

    # test = xgb.DMatrix(test)


    k_fold = KFold(n_splits=splits_nums, shuffle=True, random_state=2019)
    for index, (train_index, test_index) in enumerate(k_fold.split(train)):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = my_model(X_train, y_train, X_test, y_test)

        train_id_train, train_id_test = train_id[train_index], train_id[test_index]

        # X_test = xgb.DMatrix(X_test)
        vali_pre = model.predict(X_test, ntree_limit=model.best_iteration)
        oof[test_index] = vali_pre
        train_id_test = pd.DataFrame({'ID': train_id_test, 'score': vali_pre})
        score = r2_score(y_test, vali_pre)
        score_list.append(score)

        print(y_test)
        print(vali_pre)

        # , ntree_limit=model.best_iteration
        pred_result = model.predict(test, ntree_limit=model.best_iteration)
        sub['tradeMoney'] = pred_result

        if index == 0:
            re_sub = copy.deepcopy(sub)
            re_train = copy.deepcopy(train_id_test)
        else:
            re_sub['tradeMoney'] = re_sub['tradeMoney'] + sub['tradeMoney']
            re_train = pd.concat((re_train, train_id_test))

        # print(sub.sort_values(by=['tradeMoney']).reset_index(drop=True))

    re_sub['tradeMoney'] = re_sub['tradeMoney'] / splits_nums

    print('score list:', score_list)
    print(np.mean(score_list))
    # re_train.to_csv('../data/re_train.csv', index=None)
    return re_sub, oof

def xgb_score(y_pred, y_true):
    y_true = y_true.get_label()
    return 'my-error', 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2)

def xgb_model_bs(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(max_depth=6,
                             learning_rate=0.01,
                             objective="reg:linear",
                             eval_metric='rmse',
                             n_estimators=100000,
                             colsample_bytree=0.6,
                             reg_alpha=3,
                             reg_lambda=2,
                             gamma=0.6,
                             subsample=0.7,
                             silent=1,
                             n_jobs=-1,
                             )
    model = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=200,
                      verbose=200)

    return model

# 定义评价函数
def my_score(y_pred, y_true):
    # y_true = y_true.get_label()
    return 'my_score', 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_pred - np.mean(y_true)) ** 2), True

def lgb_model(X_train, y_train, X_test, y_test):
    x_train, y_train = X_train, y_train
    x_valid, y_valid = X_test, y_test

    clf = lgb.LGBMRegressor(
        objective='regression', num_leaves=90,
        learning_rate=0.02, n_estimators=100000, bagging_fraction=0.7,
        feature_fraction=0.6, reg_alpha=0.3, reg_lambda=0.3,
        min_data_in_leaf=18, min_sum_hessian_in_leaf=0.001, n_jobs=-1)
    clf.fit(x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=my_score,
            early_stopping_rounds=100,
            verbose=200)

    return clf


df_train = get_train(df_train)

area_money_plate = df_train['area_money'].groupby(df_train['plate']).mean().reset_index().rename(
    columns={'area_money': 'area_money_plate'})
area_money_plate_std = df_train['area_money'].groupby(df_train['plate']).std().reset_index().rename(
    columns={'area_money': 'area_money_plate_std'})
area_money_plate_min = df_train['area_money'].groupby(df_train['plate']).min().reset_index().rename(
    columns={'area_money': 'area_money_plate_min'})
area_money_plate_max = df_train['area_money'].groupby(df_train['plate']).max().reset_index().rename(
    columns={'area_money': 'area_money_plate_max'})


df = pd.concat([df_train, df_test], sort=False, axis=0, ignore_index=True)

train, test, no_features, features = get_fea(df)



X = train[features]
y = train['tradeMoney']
test_data = test[features]

sub = test[['ID']]
train_id = train['ID']

sub, oof = get_result(X, test_data, y, xgb_model_bs, need_sca=True, splits_nums=5)

sub[['tradeMoney']].to_csv('output/sub_xgb.csv', index=None, header=False)

