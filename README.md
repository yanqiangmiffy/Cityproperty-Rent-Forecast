
# 2019 未来杯高校AI挑战赛 房屋租金预测
* 战队编号：{007}
* 战队名称: {致Great}
* 战队成员：{致Great}

## 数据集

- 类别型数据
```text
categorical_feas = ['rentType', 'houseType', 'houseFloor', 'region', 'plate', 'houseToward', 'houseDecoration',
    'communityName','city','region','plate','buildYear']
```
- 数值型数据
```text
numerical_feas=['ID','area','totalFloor','saleSecHouseNum','subwayStationNum',
    'busStationNum','interSchoolNum','schoolNum','privateSchoolNum','hospitalNum',
    'drugStoreNum','gymNum','bankNum','shopNum','parkNum','mallNum','superMarketNum',
    'totalTradeMoney','totalTradeArea','tradeMeanPrice'，'tradeSecNum','totalNewTradeMoney',
    'totalNewTradeArea','tradeNewMeanPrice','tradeNewNum','remainNewNum','supplyNewNum',
    'supplyLandNum','supplyLandArea','tradeLandNum','tradeLandArea','landTotalPrice',
    'landMeanPrice','totalWorkers','newWorkers','residentPopulation','pv','uv','lookNum']
```

```python
# df['stationNum'] = df['subwayStationNum'] + df['busStationNum']
# df['schoolNum'] = df['interSchoolNum'] + df['schoolNum'] + df['privateSchoolNum']
# df['medicalNum'] = df['hospitalNum'] + df['drugStoreNum']
# df['lifeHouseNum'] = df['gymNum'] + df['bankNum'] + df['shopNum'] + df['parkNum'] + df['mallNum'] + df['superMarketNum']
# df['landSupplyTradeRatio'] = df['supplyLandArea'] / df['tradeLandArea']

# df = df.drop(['subwayStationNum', 'busStationNum',
#               'interSchoolNum', 'schoolNum', 'privateSchoolNum',
#               'hospitalNum', 'drugStoreNum',
#               'gymNum', 'bankNum', 'shopNum', 'parkNum', 'mallNum', 'superMarketNum',
#               'supplyLandArea', 'tradeLandArea'], axis=1)
```

### 结果

- lgb+使用基本特征：0.839340	
- lgb+特征工程：

### 特征工程技巧

- 数据预处理
    - 缺失数据：删除，填补(使用上下值、平均值、中位数、众数填充；根据数据之间的关联性进行填充)
    - 类别数据：one-hot编码、具有大小次序的编码
    - 数值型数据：归一化、标准化
    
- 构造特征
    - rank特征：提高鲁棒性 pd.rank()
    - k-means：聚类特征
