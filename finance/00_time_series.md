# Time Series

## 1. Handle missing data
There are three ways to handle the invalid or missing data.
- Delete the cell that contains the missing value.
- Assume that the missing cell is equal to the previous cell.
- Calculate a mean or a median of cells around the empty value.
```
import numpy as np
from sklearn import impute
# 生成维度为 (10, 6) 的随机整数矩阵 X，数值范围在 1 到 14 之间
X = np.random.randint(1, 15, (10, 6))
# 数据类型转换为 float
X = X.astype(float)
# 在矩阵 X 中随机选择 10 个位置，将这些位置的元素设置为 NaN（缺失值）
X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan
# 创建一个 KNNImputer 对象 knn_imputer，指定邻居数量为 2
knn_imputer = impute.KNNImputer(n_neighbors=2)
# # 使用 knn_imputer 对矩阵 X 进行拟合和转换，用 K-最近邻方法填补缺失值
knn_imputer.fit_transform(X)
```

## 2. Make data stationary
To transform the time series dato into stationary data, you can simply take the differences from one value relative to the previous value.

```
# Importing the required library
import pandas_datareader as pdr
# Setting the beginning and end of the historical data
start_date = '1990-01-01'
end_date = '2023-01-23'
# Creating a dataframe and downloading the VIX data
vix = pdr.DataReader('VIXCLS', 'fred', start_date, end_date)
# Printing the latest five observations of the dataframe
print(vix.tail())

# Calculating the number of nan values
count_nan = vix['VIXCLS'].isnull().sum()
# Printing the result
print('Number of nan values in the VIX dataframe: ' + str(count_nan))

# Dropping the nan values from the rows
vix = vix.dropna()

# Taking the differences in an attempt to make the data stationary
vix = vix.diff(periods = 1, axis = 0)
# Dropping the first value of the dataframe
vix = vix.iloc[1: , :]
```

## 3. Probabilistic Methods
The joint probability of the realization of two events is given by 
```
P(A, B) = P(A | B) x P(B)
```
The joint probability is given by the following formula if the two events are independent.
```
P(A, B) = P(A) x P(B)
```

The expected value of a random variable is the weighted average of the different outcomes.
```
E(X) = sum(P(x)X)
```

## 4. Sampling and Hypothesis Testing
Sampling refers to the act of selecting samples of data within a larger population and making conclusions about the statistical properties of the population. There are a few different methods of sampling.
- Simple random sampling
- Stratified sampling
- Cluster sampling
- Systematic sampling
The rule of thumb is to have a minimum of 30 observations, and the more the merrier. This brings the discussion to the central limit theorem, which states that random samples drawn from a population will approach a normal distribution as the sample gets larger.
Confidence intervals is a range of values where the population parameter is expected to be. It is generally constructed by adding or subtracting a factor from the point estimate. A confidence interval can be as follows if a sample mean x is given.
```
x ± (reliability factor x standard error)
```
The significance level is the threshold of the confidence interval. For example, a confidence interval of 95% means that with 95% confidence, the estimate should lie within a certain range. The remaining 5% probability that it does not is the significance level. (generally marked with alpha) The significence level is 5% if the confidence interval is 95%, the reliability factor is 1.96 in this case. For a 1% significance level, the reliability factor is 2.575, and for a 10% significance level, the reliability factor is 1.645.

The standard error is given by the following formula.
```
s = sigma / sqrt(n)
sigma is the population standard deviation
sqrt(n) is the square root of the population number
```

For example, Consider a population of 100 financial instruments (bonds, currency pairs, stocks, structured products, etc.). The mean annual return of these instruments is 1.4%. Assuming a population standard deviation of 4.34%, what is the confidence interval at a 1% significance level (99% confidence interval) of the mean? The answer is as follows:
```
1.4% ± 2.575 x 4.34% / sqrt(100) = 1.4% ± 1.11%
```

If the sample size is small and/or the population standard deviation is unknown, a t-distribution may be a better choice than a normal distribution.

## 5. Hypothesis Testing

In statistics, hypothesis testing is a technique for drawing conclusions about a population from a small sample of data. It entails developing two competing hypotheses, the null hypothesis and the alternative hypothesis, about a population parameter and then figuring out which is more likely to be accurate using sample data.

## 6. Entropy
Entropy is maximized when all the outcomes of a random event have the same probability. This can also be presented as a distribution, where a symmetrical distribution (such as the normal distribution) has high entropy and a skewed distribution has low entropy.