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

def slip_train_val_for_price(train, start_date, offset):
    end_date = start_date + dt.timedelta(days=offset-1)
    return train[start_date:(end_date + dt.timedelta(days=-1))], train[end_date:]

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
    data['Day'] = data.index.day
    data['DayOfWeek'] = data.index.dayofweek
    data['WeekOfYear'] = data.index.isocalendar().week
    data['DayOfYear'] = data.index.dayofyear
    data['Quarter'] = data.index.quarter
    data['JpHoliday'] = [1 if d.date() in holiday else 0 for d in data.index]

    #moving_info = pd.read_csv('ext_moving_info.csv', low_memory=False, index_col='datetime', parse_dates=True)
    #for i in data.index:
    #    for j in moving_info.index:
    #        if i.year == j.year and i.month == j.month:
    #            data.loc[i, 'y_by_move'] = np.log(moving_info.loc[j, 'y_hat'])

    #weather = pd.read_csv('weather.csv', low_memory=False, index_col='datetime', parse_dates=True)
    #for i in data.index:
    #    data.loc[i, 'rain'] = (weather.loc[i, 'saitama'] +  weather.loc[i, 'chiba'] + weather.loc[i, 'tokyo'] + weather.loc[i, 'kanagawa'])/4

    #for i in data.index:
    #    if data.loc[i, 'DayOfWeek'] == 5 or data.loc[i, 'DayOfWeek'] == 6:
    #        data.loc[i, 'JpHoliday'] = 1
    """
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
    """

    # features for time series
    #data = features_add_current_for_y(y_data, data)
    #weather = pd.read_csv("./weather.csv", low_memory=False, index_col='datetime', parse_dates=True)
    #data['kanagawa'] = weather['kanagawa']

    # features for time series
    data.sort_values(['datetime'],ascending = True, inplace=True)
    
    return data

def find_params_for_y():
    train = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)
    test = pd.read_csv("./test.csv", low_memory=False, index_col='datetime', parse_dates=True)
    train = train.astype({"y": float})
    y_train, y_val = slip_train_val_for_price(train, dt.date(2010, 7, 1), 1736)
    #y_train, y_val = slip_train_val_for_price(train, dt.date(2010, 12, 31), 1553)
    y_train = features_create_for_y(y_train, y_train)
    y_val = features_create_for_y(y_val, y_val)
    y_train.to_csv('y_train.csv', index=True, header=True)
    y_val.to_csv('y_val.csv', index=True, header=True)

    # predict price_am and price_pm
    print('*****STEP1***** create train/validation dataset for y prediction')
    y_train = pd.read_csv("./y_train.csv", low_memory=False, parse_dates=[0])
    y_val = pd.read_csv("./y_val.csv", low_memory=False, parse_dates=[0])

    print('*****STEP2***** train for y')
    import torch
    torch.set_float32_matmul_precision('high')

    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    item_list = []
    for i in range(y_train.shape[0]):
        item_list.append('H1')
    y_train['item_id'] = item_list

    train_data = TimeSeriesDataFrame.from_data_frame(
        y_train,
        id_column="item_id",
        timestamp_column="datetime"
    )
    predictor = TimeSeriesPredictor(
        prediction_length=366,
        path="autogluon-daily",
        target="y",
        eval_metric="MSE",
        known_covariates_names=['Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Quarter', 'JpHoliday', 'close', 'client', 'price_am', 'price_pm', 'y_by_move']
        #known_covariates_names=['saitama', 'chiba', 'tokyo', 'kanagawa', 'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Quarter', 'JpHoliday', 'client', 'close', 'price_am', 'price_pm'],
    ).fit(train_data,
        presets="best_quality",
        hyperparameter_tune_kwargs={
            "scheduler": "local",
            "searcher": "auto",
            "num_trials": 20,
        },
        #time_limit=600,
        #num_val_windows=1,
        enable_ensemble=True,
        refit_full=True,
        random_seed=0
    )

    item_list = []
    for i in range(y_val.shape[0]):
        item_list.append('H1')
    y_val['item_id'] = item_list
    val_data = TimeSeriesDataFrame.from_data_frame(
        y_val.drop(columns=['y']),
        id_column="item_id",
        timestamp_column="datetime"
    )
    predictor.fit_summary()
    preds_y = predictor.predict(train_data, known_covariates=val_data)
    preds_y.to_csv('try.csv', index=True, header=True)

    mae = metrics.mean_absolute_error(y_val['y'], preds_y['mean'])
    rmse = np.sqrt(metrics.mean_squared_error(y_val['y'], preds_y['mean']))
    r2 = metrics.r2_score(y_val['y'], preds_y['mean'])
    print(f"y: mae({mae}), rmse({rmse}), r2({r2})")
    """
    from autogluon.tabular import TabularDataset, TabularPredictor
    predictor_y = TabularPredictor(label='y',
                                #problem_type='regression',
                                #eval_metric='root_mean_squared_error'
                                #eval_metric='mean_absolute_error'
                                ).fit(
        train_y,
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=9,
        #refit_full=True,
        #set_best_to_refit_full=True
    )
    predictor_y.leaderboard(silent=True)
    predictor_y.fit_summary()

    print('*****STEP3***** predict y')
    preds_y = predictor_y.predict(val_y.drop(columns=['y']))
    preds_y.to_csv('preds_y.csv', index=True, header=True)
    
    preds_y = pd.read_csv("./preds_y.csv",parse_dates=[0], low_memory=False)
    mae = metrics.mean_absolute_error(val_y['y'], preds_y['y'])
    rmse = np.sqrt(metrics.mean_squared_error(val_y['y'], preds_y['y']))
    r2 = metrics.r2_score(val_y['y'], preds_y['y'])
    print(f"y: mae({mae}), rmse({rmse}), r2({r2})")
    """

def predict_y():
    y_train = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)
    y_val = pd.read_csv("./test.csv", low_memory=False, index_col='datetime', parse_dates=True)
    y_train = y_train.astype({"y": float})
    y_train = features_create_for_y(y_train, y_train)
    y_val = features_create_for_y(y_val, y_val)
    y_train.to_csv('y_train.csv', index=True, header=True)
    y_val.to_csv('y_val.csv', index=True, header=True)

    # new code for regression
    y_train = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)

    y_train, y_val = slip_train_val_for_price(y_train, dt.date(2010, 7, 1), 275+366+365+365+365)
    y_train = features_create_for_y(y_train, y_train)
    y_val = features_create_for_y(y_val, y_val)
    y_train.to_csv('y_train.csv', index=True, header=True)
    y_val.to_csv('y_val.csv', index=True, header=True)
    # predict price_am and price_pm
    print('*****STEP1***** create train/validation dataset for y prediction')
    y_train = pd.read_csv("./y_train.csv", low_memory=False, parse_dates=[0])
    y_val = pd.read_csv("./y_val.csv", low_memory=False, parse_dates=[0])
    print(y_train.head())
    print(y_train.tail())

    print('*****STEP2***** train for y')
    import torch
    torch.set_float32_matmul_precision('high')

    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    item_list = []
    for i in range(y_train.shape[0]):
        item_list.append('H1')
    y_train['item_id'] = item_list

    train_data = TimeSeriesDataFrame.from_data_frame(
        y_train,
        id_column="item_id",
        timestamp_column="datetime"
    )
    """
    predictor = TimeSeriesPredictor(
        prediction_length=365,
        path="autogluon-daily",
        target="y",
        eval_metric="MSE",
        known_covariates_names=['Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Quarter', 'JpHoliday', 'client', 'close', 'price_am', 'price_pm'],
    ).fit(train_data,
          presets="best_quality",
          hyperparameter_tune_kwargs={
            "scheduler": "local",
            "searcher": "auto",
            "num_trials": 20},
          #time_limit=600,
          #num_val_windows=1,
          enable_ensemble=True,
          refit_full=True,
          random_seed=0
          )
    """
    predictor = TimeSeriesPredictor(
        prediction_length=365,
        path="autogluon-daily",
        target="y",
        eval_metric="MSE",
        known_covariates_names=['Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'DayOfYear', 'Quarter', 'JpHoliday', 'client', 'close', 'price_am', 'price_pm'],
    ).load('autogluon-daily', require_version_match=False)

    item_list = []
    for i in range(y_val.shape[0]):
        item_list.append('H1')
    y_val['item_id'] = item_list
    val_data = TimeSeriesDataFrame.from_data_frame(
        y_val,
        id_column="item_id",
        timestamp_column="datetime"
    )
    preds_y = predictor.predict(train_data, known_covariates=val_data)
    #preds_y = predictor.predict(train_data, known_covariates=train_data)
    preds_y.to_csv('preds_y.csv', index=True, header=True)

    y_val = pd.read_csv("./y_val.csv", low_memory=False)
    y_val.drop(columns=['client','close','price_am','price_pm','Month','Day','DayOfWeek','WeekOfYear','DayOfYear','Quarter','JpHoliday'],
               inplace=True)
    preds_y = pd.read_csv("./preds_y.csv", low_memory=False)

    y_val['y'] = preds_y['mean']
    y_val.set_index('datetime', inplace=True)
    y_val.to_csv('MSE_auto20_core-Year_train2.csv', index=True, header=False)
    
if __name__ == "__main__":
    #find_params_for_y()
    predict_y()