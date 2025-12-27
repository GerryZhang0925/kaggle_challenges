# Descriptive Statistics
## 1. Retrieve CPI data
```
# Importing the required library
import pandas_datareader as pdr
# Setting the beginning and end of the historical data
start_date = '1950-01-01'
end_date = '2023-01-23'
# Creating a dataframe and downloading the CPI data
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
# Printing the latest five observations of the dataframe
print(cpi.tail())
# Checking if there are nan values in the CPI dataframe
count_nan = cpi['CPIAUCSL'].isnull().sum()
# Printing the result
print('Number of nan values in the CPI dataframe: ' + str(count_nan))
# Transforming the CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
# Dropping the nan values from the rows
cpi = cpi.dropna()
```

## 2. Measures of Central Tendency
There are three central tendency measures, mean, median and mode. It may be interesting to know that with a skewed distribution, the median may be the preferred metric since the mean tends to be pulled by outliers, thus distorting its value.
The following code shows mean in CPI calculation.
```
cpi = cpi.reset_index()
cpi_latest = cpi.iloc[–240:]
cpi_latest.set_index('DATE', inplace=True)
mean = cpi_latest["CPIAUCSL"].mean()
# Printing the result
print('The mean of the dataset: ' + str(mean), '%')
# Importing the required library
import matplotlib.pyplot as plt
# Plotting the latest observations in black with a label
plt.plot(cpi_latest["CPIAUCSL"], color = 'black', linewidth = 1.5,
label = 'Change in CPI Year-on-Year')
# Plotting horizontal lines that represent the mean and the zero threshold
plt.axhline(y = mean, color = 'red', linestyle = 'dashed',
label = 'Mean')
plt.axhline(y = 0, color = 'blue', linestyle = 'dashed', linewidth = 1)
plt.grid()
plt.legend()
```

## 3. Measures of Variability
There are three key variability metrics, the variance, the standard deviation and the range. 
The variance describes the variability of a set of numbers from their mean.
```
sigma^2 = sum(xi - mean(x))^2/n
```
The following code can be used to calculate the variance of CPI.
```
# Calculating the variance
variance = cpi_latest["CPIAUCSL"].var()
# Printing the result
print('The variance of the dataset: ' + str(variance), '%')
```

The standard deviation is the sqrt of the variance.
```
 Calculating the standard deviation
standard_deviation = cpi_latest["CPIAUCSL"].std()
# Printing the result
print('The standard deviation of the dataset: ' +
       str(standard_deviation), '%')
```
The range is given by the difference of maximum and minimum.
```
# Calculating the range
range_metric = max(cpi["CPIAUCSL"]) – min(cpi["CPIAUCSL"])
# Printing the result
print('The range of the dataset: ' + str(range_metric), '%')
```

## 4. Measures of Shape

Probability distribution, skewness, kurtosis and quantile are the focus points.

Probability distribution is a mathematical function that describes the likelihood of different outcomes or events in a random experiment.
The best-known discrete distributions are the Bernoulli distribution, binomial distribution and Poison distribution.

```
# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
# Generate data for the plot
data = np.linspace(–3, 3, num = 1000)
# Define the mean and standard deviation of the normal distribution
mean = 0
std = 1
# Generate the function of the normal distribution
pdf = stats.norm.pdf(data, mean, std)
# Plot the normal distribution plot
plt.plot(data, pdf, '-', color = 'black', lw = 2)
plt.axvline(mean, color = 'black', linestyle = '--')
plt.grid()
plt.show()
```

Skewness describes a distribution's asymmetry. The skewness of a normal distribution is equal to zero.
A positive skew indicates that the distribution has a long tail to the right, which means that the mean is greater than the median because the mean is sensible to outliers. In the world of financial markets, ff the distribution is positively skewed, it means that there are more returns above the mean than below it. The skew of a returns series can provide information about the risk and return of an investment. For example, a positively skewed returns series may indicate that the investment has a potential for a few large gains with a risk of frequent small losses.
```
# Calculating the skew
skew = cpi_latest["CPIAUCSL"].skew()
# Printing the result
print('The skew of the dataset: ' + str(skew))
```
- f skewness is between –0.5 and 0.5, the data is considered symmetrical.
- If skewness is between –1.0 and –0.5 or between 0.5 and 1.0, the data is considered mildly skewed.
- If skewness is less than –1.0 or greater than 1.0, the data is considered highly skewed.
```
# Plotting the histogram of the data
fig, ax = plt.subplots()
ax.hist(cpi['CPIAUCSL'], bins = 30, edgecolor = 'black', color = 'white')
# Add vertical lines for better interpretation
ax.axvline(mean, color='black', linestyle='--', label = 'Mean', linewidth = 2)
ax.axvline(median, color='grey', linestyle='-.', label = 'Median', linewidth = 2)
plt.grid()
plt.legend()
plt.show()
```

Kurtosis is a description of the peakedness or flatness of a distribution relative to a normal distribution. It describes the tails of a distribution.
A normal distributionhas a kurtosis of 3, which means it is a mesokuritic distribution. If a distribution has a kurtosis greater than 3, it is refered to as leptokurtic, meaning it has a higher peak and fatter tails than a normal distribution. If a distribution has a kurtosis less than 3, it is refereed to as platykyurtic, meaning it has a flatter peak and thinner tails than a normal distribution.

Sometimes kurtosis is measured as excess kutosis to give it a starting value of zero for a normal distribution.
```
# Calculating the excess kurtosis
excess_kurtosis = cpi_latest["CPIAUCSL"].kurtosis()
# Printing the result
print('The excess kurtosis of the dataset: ' + str(excess_kurtosis))
```

Quantiles divide the arranged dataset into equal parts. IQR is the difference between the third quartile and the first quartile.

## 5. Visualization
Scatterplots are used to graph the relationship between two variables through points that correspond to the intersection between the variables.
```
import matplotlib.pyplot as plt
# Resetting the index
cpi = cpi.reset_index()
# Creating the chart
fig, ax = plt.subplots()
ax.scatter(cpi['DATE'], cpi['CPIAUCSL'], color = 'black',
s = 8, label = 'Change in CPI Year-on-Year')
plt.grid()
plt.legend()
plt.show()
```
Scatterplots are more commonly used to compare variables with removing the time variable.
```
# Setting the beginning and end of the historical data
start_date = '1995-01-01'
end_date = '2022-12-01'
# Creating a dataframe and downloading the CPI data
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)
# Dropping the NaN values from the rows
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()
# Transforming the CPI into a year-on-year measure
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()
# Creating the chart
fig, ax = plt.subplots()
ax.scatter(cpi_us['CPIAUCSL'], cpi_uk['GBRCPIALLMINMEI'],
           color = 'black', s = 8, label = 'Change in CPI Year-on-Year')
# Adding a few aesthetic elements to the chart
ax.set_xlabel('US CPI')
ax.set_ylabel('UK CPI')
ax.axvline(x = 0, color='black', linestyle = 'dashed', linewidth = 1)
ax.axhline(y = 0, color='black', linestyle = 'dashed', linewidth = 1)
ax.set_ylim(-2,)
plt.grid()
plt.legend()
plt.show()
```

Histogram is a specific sort of barchart that is used to display the frequency distribution of continuous data by using bars to represent statistical information.
```
# Creating the chart
fig, ax = plt.subplots()
ax.hist(cpi['CPIAUCSL'], bins = 30, edgecolor = 'black',
color = 'white', label = 'Change in CPI Year-on-Year',)
# Add vertical lines for better interpretation
ax.axvline(0, color = 'black')
plt.grid()
plt.legend()
plt.show()
```

The box and whisker plot is used to visualize the distribution of continuous variables while including the median and the quartiles as well as the outliers.
```
# Creating the chart
cpi_latest = cpi.iloc[–240:]
fig, ax = plt.subplots()
ax.boxplot(cpi_latest['CPIAUCSL'])
plt.grid()
plt.legend()
plt.show()
```
To remove the outliers from the plot, you simply use the following tweak:
```
fig, ax = plt.subplots()
ax.boxplot(cpi_latest['CPIAUCSL'], showfliers = False)
```

## 6. Correlation

There are three main methods for calculating correlation: the Spearman method, the Pearson method and MIC.

Pearson correlation is usually used with variables that have proportional changes and are normally distributed.
```
# Importing the required libraries
import pandas_datareader as pdr
import pandas as pd
# Setting the beginning and end of the historical data
start_date = '1995-01-01'
end_date = '2022-12-01'
# Creating a dataframe and downloading the CPI data
cpi_us = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi_uk = pdr.DataReader('GBRCPIALLMINMEI', 'fred', start_date, end_date)
# Dropping the nan values from the rows
cpi_us = cpi_us.dropna()
cpi_uk = cpi_uk.dropna()
# Transforming the US CPI into a year-on-year measure
cpi_us = cpi_us.pct_change(periods = 12, axis = 0) * 100
cpi_us = cpi_us.dropna()
# Transforming the UK CPI into a year-on-year measure
cpi_uk = cpi_uk.pct_change(periods = 12, axis = 0) * 100
cpi_uk = cpi_uk.dropna()
# Joining both CPI data into one dataframe
combined_cpi_data = pd.concat([cpi_us['CPIAUCSL'],
cpi_uk['GBRCPIALLMINMEI']], axis = 1)
# Calculating Pearson correlation
combined_cpi_data.corr(method = 'pearson')
```

Spearman’s rank correlation is a nonparametric rank correlation that measures the
strength of the relationship between the variables. It is suitable for variables that do
not follow a normal distribution.

```
# Calculating Spearman's rank correlation
combined_cpi_data.corr(method = 'spearman')
```

Autocorrelation (also referred to as serial correlation) is a statistical method used to look at the relationship between a given time series and a lagged version of it.

Positive autocorrelation frequently occurs in trending assets and is associated with the idea of persistence (trend following). Negative autocorrelation is exhibited in ranging markets and is associated with the idea of antipersistence (mean reversion).

```
# Creating a dataframe and downloading the CPI data
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
# Transforming the US CPI into a year-on-year measure
cpi = cpi.pct_change(periods = 12, axis = 0) * 100
cpi = cpi.dropna()
# Transforming the data frame to a series structure
cpi = cpi.iloc[:,0]
# Calculating autocorrelation with a lag of 1
print('Correlation with a lag of 1 = ', round(cpi.autocorr(lag = 1), 2))
# Calculating autocorrelation with a lag of 6
print('Correlation with a lag of 6 = ', round(cpi.autocorr(lag = 6), 2))
# Calculating autocorrelation with a lag of 12
print('Correlation with a lag of 12 = ', round(cpi.autocorr(lag = 12), 2))
```

The maximal information coefficient (MIC) is a nonparametric measure of association between two variables designed to handle large and complex data. It is generally seen as a more robust alternative to traditional measures of correlation, such as Pearson correlation and Spearman’s rank correlation.  It can handle high-dimensional data and can identify nonlinear relationships between variables. however, nondirectional, which means that values close to 1 only suggest a strong correlation between the two variables.

The following code shows the different result with the three methods.
```
# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
# Setting the range of the data
data_range = np.arange(0, 30, 0.1)
# Creating the sine and the cosine waves
sine = np.sin(data_range)
cosine = np.cos(data_range)
# Plotting
plt.plot(sine, color = 'black', label = 'Sine Function')
plt.plot(cosine, color = 'grey', linestyle = 'dashed',
label = 'Cosine Function')
plt.grid()
plt.legend()

# Importing the library
from minepy import MINE
# Calculating the MIC
mine = MINE(alpha = 0.6, c = 15)
mine.compute_score(sine, cosine)
MIC = mine.mic()
print('Correlation | MIC: ', round(MIC, 3))
```
The result is
```
Correlation | Pearson: 0.035
Correlation | Spearman: 0.027
Correlation | MIC: 0.602
```

## 7. Stationarity

Stationarity occurs when the statistical characteristics of the time series (mean, variance, etc.) are constant over time. 
In other words, no discernable trend is detectable when plotting the data across time.

Stationarity can be checked with two methods, augmented Dickey-Fuller test and Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test.

### 7.1 augmented Dickey-Fuller test

```
# Importing the required library
from statsmodels.tsa.stattools import adfuller
# Applying the ADF test on the CPI data
print('p-value: %f' % adfuller(cpi)[1])
```
Its result is
```
p-value: 0.0152
```
Assuming a 5% significance level, it seems that it is possible to accept that the year-on-year data is stationary (however, if you want to be stricter and use a 1% signifi‐cance level, then the p-value suggests that the data is nonstationary).

But on the raw CPI data, 
```
# Creating a dataframe and downloading the CPI data
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
# Applying the ADF test on the CPI data
print('p-value: %f' % adfuller(cpi)[1])
```
It is nonstationary.
```
p-value: 0.999
```

### 7.2 Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test
```
# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
# Creating the first time series using sine waves
length = np.pi * 2 * 5
sinewave = np.sin(np.arange(0, length, length / 1000))
# Creating the second time series using trending sine waves
sinewave_ascending = np.sin(np.arange(0, length, length / 1000))
# Defining the trend variable
a = 0.01
# Looping to add a trend factor
for i in range(len(sinewave_ascending)):
    sinewave_ascending[i] = a + sinewave_ascending[i]
a = 0.01 + a

# Plotting the series
plt.plot(sinewave, label = 'Sine Wave', color = 'black')
plt.plot(sinewave_ascending, label = 'Ascending Sine Wave',
color = 'grey')
plt.grid()
plt.legend()
plt.show()

# Importing the KPSS library
from statsmodels.tsa.stattools import kpss
# KPSS testing | Normal sine wave
print('p-value: %f' % kpss(sinewave)[1])
# KPSS testing | Ascending sine wave
print('p-value: %f' % kpss(sinewave_ascending)[1])
# KPSS testing while taking into account the trend | Ascending sine wave
print('p-value: %f' % kpss(sinewave_ascending, regression = 'ct')[1])
```
The output is as follows:
```
p-value: 0.10 # For the sine wave
p-value: 0.01 # For the ascending sine wave without trend consideration
p-value: 0.10 # For the ascending sine wave with trend consideration
```
The KPSS statistic, when taking into account the trend, states that the ascending sine wave is a stationary time series. 
The ADF and KPSS tests check for stationarity in the data, with the latter being able to check for stationarity in trending data.
Trending data may be stationary. Although this characteristic is rare, the KPSS test can detect the stationarity, while the ADF test cannot.