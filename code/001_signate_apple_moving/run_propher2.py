# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from time import time
import jpholiday
import datetime as dt
import sklearn.metrics as metrics
from dateutil.relativedelta import relativedelta
from scipy.stats import skew
from scipy.stats import kurtosis
import itertools
from prophet.diagnostics import cross_validation

def features_create_for_y(y_data, src):  
    holiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    holiday = [x[0] for x in holiday]
    data = src.copy()
    #tiansha = [dt.date(2010, 8, 26), dt.date(2010, 10, 25), dt.date(2010, 11, 10),
    #           dt.date(2011, 1, 9), dt.date(2011, 3, 24), dt.date(2011, 3, 24),
    #           dt.date(2011, 6, 8)]

    # normal features
    #data['Year'] = data.index.year
    data['Month'] = data.index.month
    #data['Day'] = data.index.day
    #data['DayOfWeek'] = data.index.dayofweek
    data['WeekOfYear'] = data.index.isocalendar().week
    data['DayOfYear'] = data.index.dayofyear
    #data['Quarter'] = data.index.quarter
    #data['JpHoliday'] = [1 if d.date() in holiday else 0 for d in data.index]

    #for i in data.index:
    #    if data.loc[i, 'DayOfWeek'] == 5 or data.loc[i, 'DayOfWeek'] == 6:
    #        data.loc[i, 'JpHoliday'] = 1

    from qreki import Kyureki
    for i in data.index:
        k = Kyureki.from_date(i)
        if (k.rokuyou == '大安'):
            data.loc[i, 'Kyureki'] = 1
        elif (k.rokuyou == '赤口'):
            data.loc[i, 'Kyureki'] = 2
        elif (k.rokuyou == '先勝'):
            data.loc[i, 'Kyureki'] = 3
        elif (k.rokuyou == '友引'):
            data.loc[i, 'Kyureki'] = 4
        elif (k.rokuyou == '先負'):
            data.loc[i, 'Kyureki'] = 5
        else: # 仏滅
            data.loc[i, 'Kyureki'] = 0

    moving_info = pd.read_csv('moving_info.csv', low_memory=False, index_col='datetime', parse_dates=True)
    for i in data.index:
        for j in moving_info.index:
            if i.year == j.year and i.month == j.month:
                data.loc[i, 'saitama'] = np.log(moving_info.loc[j, 'saitama'])
                data.loc[i, 'chiba'] = np.log(moving_info.loc[j, 'chiba'])
                data.loc[i, 'tokyo'] = np.log(moving_info.loc[j, 'tokyo'])
                data.loc[i, 'kanagawa'] = np.log(moving_info.loc[j, 'kanagawa'])
                data.loc[i, 'osaka'] = np.log(moving_info.loc[j, 'osaka'])

    # features for time series
    #data = features_add_current_for_y(y_data, data)
    #weather = pd.read_csv("./weather.csv", low_memory=False, index_col='datetime', parse_dates=True)
    #data['saitama'] = weather['saitama']
    #data['kanagawa'] = weather['kanagawa']
    #data['tokyo'] = weather['tokyo']
    #data['chiba'] = weather['chiba']
    # features for time series
    data.sort_values(['datetime'],ascending = True, inplace=True)
    
    return data

from prophet import Prophet

def tune_params():
    train = pd.read_csv("./train_repaired.csv", low_memory=False, parse_dates=[0])
    test = pd.read_csv("./test.csv", low_memory=False, parse_dates=[0])
    train = train.astype({"y": float})
    train.set_index('datetime', inplace=True)
    test.set_index('datetime', inplace=True)
    train = features_create_for_y(train, train)
    test = features_create_for_y(train, test)

    import jpholiday
    
    jholiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    jholiday = [x[0] for x in jholiday]
    holidays = []
    holiday_names = []
    for i in train.index:
        if i in jholiday:
            holidays.append(i)
            holiday_names.append('jpholiday')
        elif i.dayofweek == 5 or i.dayofweek == 6:
            holidays.append(i)
            holiday_names.append('weekend')
        elif i.month == 3 and i.day > 10:
            holidays.append(i)
            holiday_names.append('match')
        elif i.month == 9 and i.day > 10:
            holidays.append(i)
            holiday_names.append('september')
    holidays = pd.DataFrame({
        'holiday': holiday_names,
        'ds': holidays,
        'lower_window': 0,
        'upper_window': 1,
    })

    train.reset_index(inplace=True)
    train = train.rename(columns={'datetime': 'ds'})
    train['cap']=95
    train['floor']=0

    # fit_model
    m=Prophet(
        holidays=holidays,
        holidays_prior_scale=6.6, #祝日効果の強さをいじるパラメータ
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth = 'logistic',
        changepoint_prior_scale=0.181,
        changepoint_range=0.822,
        n_changepoints=25,
        seasonality_prior_scale=24.5, 
        seasonality_mode='multiplicative'
      )
    m.add_seasonality(name='yearly', period=365.25, fourier_order=18,prior_scale=16.501, mode='multiplicative')
    #m.add_seasonality(name='monthly', period=28.5, fourier_order=4,prior_scale=0.001, mode='multiplicative')
    m.add_seasonality(name='weekly', period=7, fourier_order=4,prior_scale=19.001)
    m.add_seasonality(name='quaterly', period=365.25/4, fourier_order=9,prior_scale=3.501)
    m.add_regressor(name='client', prior_scale=10.0)
    m.add_regressor(name='close', prior_scale=14.0)
    m.add_regressor(name='price_am', prior_scale=5.501)
    m.add_regressor(name='price_pm', prior_scale=13.001)
    m.add_regressor(name='Month', prior_scale=46.5)
    m.add_regressor(name='DayOfYear', prior_scale=33.5)
    m.add_regressor(name='Kyureki', prior_scale=20.5)
    m.add_regressor(name='saitama', prior_scale=16.0)
    m.add_regressor(name='chiba', prior_scale=9.0)
    m.add_regressor(name='tokyo', prior_scale=22.0)
    m.add_regressor(name='kanagawa', prior_scale=7.5)
    m.fit(train)
    
    df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
    from prophet.diagnostics import performance_metrics
    #df_cv = performance_metrics(df_cv)
    #df_cv.to_csv('valid_result.csv', index=True, header=True)
    from prophet.plot import plot_cross_validation_metric
    plot_cross_validation_metric(df_cv, metric='mae')


if __name__ == "__main__":
    tune_params()