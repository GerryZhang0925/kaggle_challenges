- [Feature Engineering](#feature-engineering)
  - [1. Techniques for Numerical Features](#1-techniques-for-numerical-features)
    - [1.1 Feature creation directly from date information](#11-feature-creation-directly-from-date-information)
    - [1.2 Apply aggregation calculation to date information](#12-apply-aggregation-calculation-to-date-information)
    - [1.3 Apply statistical calculation to date information](#13-apply-statistical-calculation-to-date-information)
    - [1.4 Features with tsfresh library](#14-features-with-tsfresh-library)
    - [1.5 Features with scikit-learn library](#15-features-with-scikit-learn-library)
    - [1.6 Transform numerical features into counts of categorical features](#16-transform-numerical-features-into-counts-of-categorical-features)
    - [1.7 Reduced value of the existing features](#17-reduced-value-of-the-existing-features)

# Feature Engineering
Feature engineering includes feature creation, normalization and transformation.

## 1. Techniques for Numerical Features

Features with small variance can be removed because they are same as constants. We can use VarianceThreshold of Scikit-learn to remove such kind of features.

## 1.1 Feature creation directly from date information

It is easy to handle datetime information with pandas.
```
# 添加'year'列，将 'datetime_column' 中的年份提取出来
df.loc[:, 'year'] = df['datetime_column'].dt.year
# 添加'weekofyear'列，将 'datetime_column' 中的周数提取出来
df.loc[:, 'weekofyear'] = df['datetime_column'].dt.weekofyear
# 添加'month'列，将 'datetime_column' 中的月份提取出来
df.loc[:, 'month'] = df['datetime_column'].dt.month
# 添加'dayofweek'列，将 'datetime_column' 中的星期几提取出来
df.loc[:, 'dayofweek'] = df['datetime_column'].dt.dayofweek
# 添加'weekend'列，判断当天是否为周末
df.loc[:, 'weekend'] = (df.datetime_column.dt.weekday =5).astype(int)
# 添加 'hour' 列，将 'datetime_column' 中的小时提取出来
df.loc[:, 'hour'] = df['datetime_column'].dt.hour

import pandas as pd
# 创建日期时间序列，包含了从 '2020-01-06' 到 '2020-01-10' 的日期时间点，时间间隔为10小时
s = pd.date_range('2020-01-06', '2020-01-10', freq='10H').to_series()
# 提取对应时间特征
features = {
    "dayofweek": s.dt.dayofweek.values,
    "dayofyear": s.dt.dayofyear.values,
    "hour": s.dt.hour.values,
    "is_leap_year": s.dt.is_leap_year.values,
    "quarter": s.dt.quarter.values,
    "weekofyear": s.dt.weekofyear.values
}
```

## 1.2 Apply aggregation calculation to date information
```
def generate_features(df):
    df.loc[:, 'year'] = df['date'].dt.year
    df.loc[:, 'weekofyear'] = df['date'].dt.weekofyear
    df.loc[:, 'month'] = df['date'].dt.month
    df.loc[:, 'dayofweek'] = df['date'].dt.dayofweek
    df.loc[:, 'weekend'] = (df['date'].dt.weekday =5).astype(int)
    aggs = {}
    # 对 'month' 列进行 nunique 和 mean 聚合
    aggs['month'] = ['nunique', 'mean']
    # 对 'weekofyear' 列进行 nunique 和 mean 聚合
    aggs['weekofyear'] = ['nunique', 'mean']
    # 对 'num1' 列进行 sum、max、min、mean 聚合
    aggs['num1'] = ['sum','max','min','mean']
    # 对 'customer_id' 列进行 size 聚合
    aggs['customer_id'] = ['size']
    # 对 'customer_id' 列进行 nunique 聚合
    aggs['customer_id'] = ['nunique']
    # 对数据应用不同的聚合函数
    agg_df = df.groupby('customer_id').agg(aggs)
    # 重置索引
    agg_df = agg_df.reset_index()
    return agg_df
```

## 1.3 Apply statistical calculation to date information
```
import numpy as np
# 创建字典，用于存储不同的统计特征
feature_dict = {}
# 计算 x 中元素的平均值，并将结果存储在 feature_dict 中的 'mean' 键下
feature_dict['mean'] = np.mean(x)
# 计算 x 中元素的最大值，并将结果存储在 feature_dict 中的 'max' 键下
feature_dict['max'] = np.max(x)
# 计算 x 中元素的最小值，并将结果存储在 feature_dict 中的 'min' 键下
feature_dict['min'] = np.min(x)
# 计算 x 中元素的标准差，并将结果存储在 feature_dict 中的 'std' 键下
feature_dict['std'] = np.std(x)
# 计算 x 中元素的方差，并将结果存储在 feature_dict 中的 'var' 键下
feature_dict['var'] = np.var(x)
# 计算 x 中元素的差值，并将结果存储在 feature_dict 中的 'ptp' 键下
feature_dict['ptp'] = np.ptp(x)
# 计算 x 中元素的第10百分位数（即百分之10分位数），并将结果存储在 feature_dict 中的 'percentile_10' 键下
feature_dict['percentile_10'] = np.percentile(x, 10)
# 计算 x 中元素的第60百分位数，将结果存储在 feature_dict 中的 'percentile_60' 键下
feature_dict['percentile_60'] = np.percentile(x, 60)
# 计算 x 中元素的第90百分位数，将结果存储在 feature_dict 中的 'percentile_90' 键下
feature_dict['percentile_90'] = np.percentile(x, 90)
# 计算 x 中元素的5%分位数（即0.05分位数），将结果存储在 feature_dict 中的 'quantile_5' 键下
feature_dict['quantile_5'] = np.quantile(x, 0.05)
# 计算 x 中元素的95%分位数（即0.95分位数），将结果存储在 feature_dict 中的 'quantile_95' 键下
feature_dict['quantile_95'] = np.quantile(x, 0.95)
# 计算 x 中元素的99%分位数（即0.99分位数），将结果存储在 feature_dict 中的 'quantile_99' 键下
feature_dict['quantile_99'] = np.quantile(x, 0.99)
```

## 1.4 Features with tsfresh library
```
from tsfresh.feature_extraction import feature_calculators as fc
# 计算 x 数列的绝对能量（abs_energy），并将结果存储在 feature_dict 字典中的'abs_energy' 键下
feature_dict['abs_energy'] = fc.abs_energy(x)
# 计算 x 数列中高于均值的数据点数量，将结果存储在 feature_dict 字典中的'count_above_mean' 键下
feature_dict['count_above_mean'] = fc.count_above_mean(x)
# 计算 x 数列中低于均值的数据点数量，将结果存储在 feature_dict 字典中的'count_below_mean' 键下
feature_dict['count_below_mean'] = fc.count_below_mean(x)
# 计算 x 数列的均值绝对变化（mean_abs_change），并将结果存储在 feature_dict 字典中的'mean_abs_change' 键下
feature_dict['mean_abs_change'] = fc.mean_abs_change(x)
# 计算 x 数列的均值变化率（mean_change），并将结果存储在 feature_dict 字典中的'mean_change' 键下
feature_dict['mean_change'] = fc.mean_change(x)
```

## 1.5 Features with scikit-learn library
```
import numpy as np
df = pd.DataFrame(
    np.random.rand(100, 2),
    columns=[f"f_{i}" for i in range(1, 3)])

from sklearn import preprocessing
# 指定多项式的次数为 2，不仅考虑交互项，不包括偏差（include_bias=False）
pf = preprocessing.PolynomialFeatures(degree=2,
                                    interaction_only=False,
                                    include_bias=False)
# 拟合，创建多项式特征
pf.fit(df)
# 转换数据
poly_feats = pf.transform(df)

# 获取生成的多项式特征的数量
num_feats = poly_feats.shape[1]
# 为新生成的特征命名
df_transformed = pd.DataFrame(poly_feats,columns=[f"f_{i}" for i in range(1,
    num_feats + 1)] )
```

## 1.6 Transform numerical features into counts of categorical features
```
# 创建10个分箱
df["f_bin_10"] = pd.cut(df["f_1"], bins=10, labels=False)
# 创建100个分箱
df["f_bin_100"] = pd.cut(df["f_1"], bins=100, labels=False)
```

## 1.7 Reduced value of the existing features
```
In [X]: df.f_3.var()
Out[X]: 8077265.875858586
In [X]: df.f_3.apply(lambda x: np.log(1 + x)).var()
Out[X]: 0.6058771732119975
```

Normalization is not necessary for tree-based models.
