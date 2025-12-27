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

    data['busy'] = 0
    for i in data.index:
        if i.month == 3 and i.day > 15:
            data.loc[i, 'busy'] = 1
        elif i.month == 9 and i.day > 22:
            data.loc[i, 'busy'] = 1
        elif i.month == 4 and i.day < 2:
            data.loc[i, 'busy'] = 1
        elif i.month == 10 and i.day < 3:
            data.loc[i, 'busy'] = 1
        elif i.month == 12 and i.day > 25:
            data.loc[i, 'busy'] = 1
        
    moving_info = pd.read_csv('ext_moving_info.csv', low_memory=False, index_col='datetime', parse_dates=True)
    for i in data.index:
        for j in moving_info.index:
            if i.year == j.year and i.month == j.month:
                data.loc[i, 'y_by_move'] = np.log(moving_info.loc[j, 'y_hat'])

    weather = pd.read_csv('weather.csv', low_memory=False, index_col='datetime', parse_dates=True)
    for i in data.index:
        data.loc[i, 'rain'] = (weather.loc[i, 'saitama'] +  weather.loc[i, 'chiba'] + weather.loc[i, 'tokyo'] + weather.loc[i, 'kanagawa'])/4
    #moving_info = pd.read_csv('moving_info.csv', low_memory=False, index_col='datetime', parse_dates=True)
    #for i in data.index:
    #    for j in moving_info.index:
    #        if i.year == j.year and i.month == j.month:
    #            data.loc[i, 'saitama'] = np.log(moving_info.loc[j, 'saitama'])
    #            data.loc[i, 'chiba'] = np.log(moving_info.loc[j, 'chiba'])
    #            data.loc[i, 'tokyo'] = np.log(moving_info.loc[j, 'tokyo'])
    #            data.loc[i, 'kanagawa'] = np.log(moving_info.loc[j, 'kanagawa'])
    #            data.loc[i, 'osaka'] = np.log(moving_info.loc[j, 'osaka'])

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

def check_params():
    # Prophetのモデルを「乗法」で作成する
    m = Prophet(seasonality_mode='multiplicative')

    # 追加情報
    m.add_regressor('client')
    m.add_regressor('close')
    m.add_regressor('price_am')
    m.add_regressor('price_pm')
    m.add_regressor('Month')
    #m.add_regressor('Day')
    #m.add_regressor('DayOfWeek')
    #m.add_regressor('WeekOfYear')
    m.add_regressor('DayOfYear')
    #m.add_regressor('Quarter')
    #m.add_regressor('JpHoliday')
    m.add_regressor('Kyureki')
    #m.add_regressor('saitama')
    #m.add_regressor('chiba')
    #m.add_regressor('tokyo')
    #m.add_regressor('kanagawa')
    #m.add_regressor('osaka')
    m.add_regressor('busy')
    m.add_regressor('rain')

    # 休日情報の追加。Prophet(holidays=holidays)みたいに独自の休日も定義できる
    m.add_country_holidays(country_name='JP')

    # 月次変動の追加
    #m.add_seasonality(name='yearly', period=365, fourier_order=5)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
    #m.add_seasonality(name='quartly', period=120, fourier_order=1, prior_scale=0.05)

    import pandas as pd
    from matplotlib import pyplot as plt

    train = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)
    test = pd.read_csv("./test.csv", low_memory=False, index_col='datetime', parse_dates=True)
    train = train.astype({"y": float})
    y_train, y_val = slip_train_val_for_price(train, dt.date(2010, 7, 1), 1736)
    y_train = features_create_for_y(y_train, y_train)
    y_val = features_create_for_y(y_val, y_val)
    y_train.to_csv('y_train.csv', index=True, header=True)
    y_val.to_csv('y_val.csv', index=True, header=True)

    # predict price_am and price_pm
    print('*****STEP1***** create train/validation dataset for y prediction')
    y_train = pd.read_csv("./y_train.csv", low_memory=False, parse_dates=[0])
    y_val = pd.read_csv("./y_val.csv", low_memory=False, parse_dates=[0])

    y_train = y_train.rename(columns={'datetime': 'ds'})
    y_val = y_val.rename(columns={'datetime': 'ds'})

    #y_val.drop(columns=['chiba'], inplace=True)

    # 学習する
    m.fit(y_train)
    preds_y = m.predict(y_val)

    mae = metrics.mean_absolute_error(y_val['y'], preds_y['yhat'])
    rmse = np.sqrt(metrics.mean_squared_error(y_val['y'], preds_y['yhat']))
    r2 = metrics.r2_score(y_val['y'], preds_y['yhat'])
    print(f"y: mae({mae}), rmse({rmse}), r2({r2})")
    preds_y.to_csv('propher.csv', index=True, header=False)


def objective_variable(train,valid,holidays):

    cap = int(np.percentile(train.y,95))
    floor = int(np.percentile(train.y,5))

    def objective(trial):
            params = {
                    'holidays_prior_scale': trial.suggest_discrete_uniform('holidays_prior_scale',3.5,10.0,0.1),
                    'changepoint_range' : trial.suggest_discrete_uniform('changepoint_range',0.1,1.0,0.001),
                    'n_changepoints' : trial.suggest_int('n_changepoints',12,35),
                    'changepoint_prior_scale' : trial.suggest_discrete_uniform('changepoint_prior_scale',0.1,0.8,0.001),
                    'seasonality_prior_scale' : trial.suggest_discrete_uniform('seasonality_prior_scale',20,40,0.5),
                    'threeyearly_fourier' : trial.suggest_int('threeyearly_fourier',5,30),
                    'yearly_fourier' : trial.suggest_int('yearly_fourier',5,30),
                    'monthly_fourier' : trial.suggest_int('monthly_fourier',1,15),
                    'weekly_fourier' : trial.suggest_int('weekly_fourier',1,10),
                    'quaterly_fourier' : trial.suggest_int('quaterly_fourier',3,15),
                    'threeyearly_prior' : trial.suggest_discrete_uniform('threeyearly_prior',0.001,28,0.5),
                    'yearly_prior' : trial.suggest_discrete_uniform('yearly_prior',0.001,28,0.5),
                    'monthly_prior' : trial.suggest_discrete_uniform('monthly_prior',0.001,25,0.5),
                    'weekly_prior' : trial.suggest_discrete_uniform('weekly_prior',0.001,30,0.5),
                    'quaterly_prior' : trial.suggest_discrete_uniform('quaterly_prior',0.001,30,0.5),
                    'client_prior' : trial.suggest_discrete_uniform('client_prior',10.0,55,0.5),
                    'close_prior' : trial.suggest_discrete_uniform('close_prior',10.0,30,0.5),
                    'price_am_prior' : trial.suggest_discrete_uniform('price_am_prior',0.001,30,0.5),
                    'price_pm_prior' : trial.suggest_discrete_uniform('price_pm_prior',0.001,30,0.5),
                    'month_prior' : trial.suggest_discrete_uniform('month_prior',10.0,70,0.5),
                    'dayofyear_prior' : trial.suggest_discrete_uniform('dayofyear_prior',10.0,40,0.5),
                    'kyureki_prior' : trial.suggest_discrete_uniform('kyureki_prior',10.0,30,0.5),
                    'busy_prior' : trial.suggest_discrete_uniform('busy_prior',1.0,30,0.5),
                    'rain_prior' : trial.suggest_discrete_uniform('rain_prior',1.0,30,0.5),
                    'y_by_move_prior' : trial.suggest_discrete_uniform('y_by_move_prior',1.0,30,0.5),
                    #'saitama_prior' : trial.suggest_discrete_uniform('saitama_prior',1.0,30,0.5),
                    #'chiba_prior' : trial.suggest_discrete_uniform('chiba_prior',1.0,40,0.5),
                    #'tokyo_prior' : trial.suggest_discrete_uniform('tokyo_prior',1.0,40,0.5),
                    #'kanagawa_prior' : trial.suggest_discrete_uniform('kanagawa_prior',1.0,40,0.5)
            }
            # fit_model
            m=Prophet(
                    holidays=holidays,
                    holidays_prior_scale = params['holidays_prior_scale'],
                    changepoint_range = params['changepoint_range'],
                    n_changepoints=params['n_changepoints'],
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale = params['seasonality_prior_scale'],
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    #growth='logistic',
                    seasonality_mode='multiplicative'
                    )
            m.add_seasonality(name='threeyearly', period=365*3, fourier_order=params['threeyearly_fourier'],prior_scale=params['threeyearly_prior'], mode='multiplicative')
            m.add_seasonality(name='yearly', period=365, fourier_order=params['yearly_fourier'],prior_scale=params['yearly_prior'])
            m.add_seasonality(name='monthly', period=28, fourier_order=params['monthly_fourier'],prior_scale=params['monthly_prior'], mode='multiplicative')
            m.add_seasonality(name='weekly', period=7, fourier_order=params['weekly_fourier'],prior_scale=params['weekly_prior'], mode='multiplicative')
            m.add_seasonality(name='quaterly', period=365/4, fourier_order=params['quaterly_fourier'],prior_scale=params['quaterly_prior'], mode='multiplicative')

            m.add_regressor(name='client', prior_scale=params['client_prior'])
            m.add_regressor(name='close', prior_scale=params['close_prior'])
            m.add_regressor(name='price_am', prior_scale=params['price_am_prior'])
            m.add_regressor(name='price_pm', prior_scale=params['price_pm_prior'])
            m.add_regressor(name='Month', prior_scale=params['month_prior'])
            m.add_regressor(name='DayOfYear', prior_scale=params['dayofyear_prior'])
            m.add_regressor(name='Kyureki', prior_scale=params['kyureki_prior'])
            m.add_regressor(name='busy', prior_scale=params['busy_prior'])
            m.add_regressor(name='rain', prior_scale=params['rain_prior'])
            m.add_regressor(name='y_by_move', prior_scale=params['y_by_move_prior'])
            
            #m.add_regressor(name='saitama', prior_scale=params['saitama_prior'])
            #m.add_regressor(name='chiba', prior_scale=params['chiba_prior'])
            #m.add_regressor(name='tokyo', prior_scale=params['tokyo_prior'])
            #m.add_regressor(name='kanagawa', prior_scale=params['kanagawa_prior'])

            m.fit(train)
            #future = m.make_future_dataframe(periods=len(valid))

            #forecast = m.predict(valid)
            #valid_forecast = forecast.tail(len(valid))

            #from sklearn.metrics import mean_squared_error as MSE
            #mse = MSE(valid_forecast.yhat, valid.y)
            #val_mape = np.mean(np.abs((valid_forecast.yhat-valid.y)/valid.y))*100
            #from sklearn.metrics import mean_absolute_error as MAE
            #mae = MAE(valid.y, forecast.yhat)
            #return mae
            from prophet.diagnostics import cross_validation
            df_cv = cross_validation(m, initial='1095 days', period='30 days', horizon = '365 days', parallel="processes")
            from prophet.diagnostics import performance_metrics
            df_p = performance_metrics(df_cv, rolling_window=1)
            return df_p['mae'].values[0]

    return objective

import optuna

def optuna_parameter(train,valid,holidays):
    #study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective_variable(train,valid,holidays), timeout=10000)
    optuna_best_params = study.best_params

    return study

def get_capfloor(data):
    data.set_index(['ds'], inplace=True)
    tmp = data.copy()
    tmp.reset_index(inplace=True)
    d = pd.DataFrame(tmp['ds'])
    d['y'] = tmp['y']
    d.set_index(['ds'], inplace=True)
    month_min = d.resample('M').min()
    month_max = d.resample('M').max()
    for i in data.index:
        for j in month_min.index:
            if i.year == j.year and i.month == j.month:
                max_v = month_max.loc[j, 'y']
                min_v = month_min.loc[j, 'y']
                break
        #data.loc[i, 'cap'] = 80
        #data.loc[i, 'floor'] = 0
    data.reset_index(inplace=True)
    return data

def tune_params():
    print('*****STEP1***** create train/validation dataset for y prediction')
    y_train = pd.read_csv("./y_train.csv", low_memory=False, parse_dates=[0])
    y_val = pd.read_csv("./y_val.csv", low_memory=False, parse_dates=[0])
    train_ord = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)

    train_ord = features_create_for_y(train_ord, train_ord)

    y_train = y_train.rename(columns={'datetime': 'ds'})
    y_val = y_val.rename(columns={'datetime': 'ds'})

    import jpholiday
    
    jholiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    jholiday = [x[0] for x in jholiday]
    holidays = []
    holiday_names = []
    for i in train_ord.index:
        if i in jholiday:
            holidays.append(i)
            holiday_names.append('jpholiday')
        elif i.dayofweek == 5 or i.dayofweek == 6:
            holidays.append(i)
            holiday_names.append('weekend')
    holidays = pd.DataFrame({
        'holiday': holiday_names,
        'ds': holidays,
        'lower_window': 0,
        'upper_window': 1,
    })

    #y_train = get_capfloor(y_train)
    #y_val = get_capfloor(y_val)
    #study = optuna_parameter(y_train,y_val,holidays)

    train_ord.reset_index(inplace=True)
    train_ord = train_ord.rename(columns={'datetime': 'ds'})
    study = optuna_parameter(train_ord,y_val,holidays)


    # fit_model
    m=Prophet(
        holidays=holidays,
        holidays_prior_scale = study.best_params['holidays_prior_scale'],
        changepoint_range = study.best_params['changepoint_range'],
        n_changepoints=study.best_params['n_changepoints'],
        seasonality_prior_scale = study.best_params['seasonality_prior_scale'],
        changepoint_prior_scale=study.best_params['changepoint_prior_scale'],
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        #growth='logistic',
        seasonality_mode='multiplicative'
        )
    #m.add_seasonality(name='threeyearly', period=365*3, fourier_order=study.best_params['threeyearly_fourier'],prior_scale=study.best_params['threeyearly_prior'], mode='multiplicative')
    m.add_seasonality(name='yearly', period=365, fourier_order=study.best_params['yearly_fourier'],prior_scale=study.best_params['yearly_prior'])
    m.add_seasonality(name='monthly', period=28, fourier_order=study.best_params['monthly_fourier'],prior_scale=study.best_params['monthly_prior'], mode='multiplicative')
    m.add_seasonality(name='weekly', period=7, fourier_order=study.best_params['weekly_fourier'],prior_scale=study.best_params['weekly_prior'], mode='multiplicative')
    m.add_seasonality(name='quaterly', period=365/4, fourier_order=study.best_params['quaterly_fourier'],prior_scale=study.best_params['quaterly_prior'], mode='multiplicative')

    # 追加情報
    m.add_regressor(name='client', prior_scale=study.best_params['client_prior'])
    m.add_regressor(name='close', prior_scale=study.best_params['close_prior'])
    m.add_regressor(name='price_am', prior_scale=study.best_params['price_am_prior'])
    m.add_regressor(name='price_pm', prior_scale=study.best_params['price_pm_prior'])
    m.add_regressor(name='Month', prior_scale=study.best_params['month_prior'])
    #m.add_regressor('Day')
    #m.add_regressor('DayOfWeek')
    #m.add_regressor('WeekOfYear')
    m.add_regressor(name='DayOfYear', prior_scale=study.best_params['dayofyear_prior'])
    #m.add_regressor('Quarter')
    #m.add_regressor('JpHoliday')
    m.add_regressor(name='Kyureki', prior_scale=study.best_params['kyureki_prior'])
    m.add_regressor(name='busy', prior_scale=study.best_params['busy_prior'])
    #m.add_regressor(name='rain', prior_scale=study.best_params['rain_prior'])
    #m.add_regressor(name='y_by_move', prior_scale=study.best_params['y_by_move_prior'])
    
    #m.add_regressor(name='saitama', prior_scale=study.best_params['saitama_prior'])
    #m.add_regressor(name='chiba', prior_scale=study.best_params['chiba_prior'])
    #m.add_regressor(name='tokyo', prior_scale=study.best_params['tokyo_prior'])
    #m.add_regressor(name='kanagawa', prior_scale=study.best_params['kanagawa_prior'])
    #m.add_regressor('osaka')

    # 休日情報の追加。Prophet(holidays=holidays)みたいに独自の休日も定義できる
    #m.add_country_holidays(country_name='JP')
    print(study.best_params)

    # 学習する
    m.fit(y_train)
    preds_y = m.predict(y_val)
    #print(f"train cap: {int(np.percentile(y_train.y,95))}, train floor: {int(np.percentile(y_train.y,5))}")
    #print(f"train ord cap: {int(np.percentile(train_ord.y,95))}, train ord floor: {int(np.percentile(train_ord.y,5))}")

    mae = metrics.mean_absolute_error(y_val['y'], preds_y['yhat'])
    rmse = np.sqrt(metrics.mean_squared_error(y_val['y'], preds_y['yhat']))
    r2 = metrics.r2_score(y_val['y'], preds_y['yhat'])
    print(f"y: mae({mae}), rmse({rmse}), r2({r2})")
    preds_y.to_csv('propher.csv', index=True, header=False)
    train_ord.to_csv('yy_train.csv', index=True, header=True)

def tune_params_grid():

    param_grid = {
        'holidays_prior_scale': [1.0, 5.0, 8.0, 10.0],
        'changepoint_range': [0.1, 0.5, 0.7, 0.9],
        'n_changepoints': [1, 7, 25, 34],
        'changepoint_prior_scale': [0.1, 0.5, 0.9],
        #'seasonality_mode': ['multiplicative', 'additive'],
        'quaterly_fourier': [1, 10, 20],
        #'quaterly_prior_scale': [0.005, 0.5, 20, 40],
        'monthly_fourier': [1, 10, 20],
        #'monthly_prior_scale': [0.005, 0.5, 20, 40],
        'weekly_fourier': [1, 10, 20],
        #'weekly_prior_scale': [0.005, 0.5, 20, 40],
        'yearly_fourier': [1, 15, 30],
        #'yearly_prior_scale': [0.005, 0.5, 20, 40],
        'threeyearly_fourier': [1, 10, 20, 30],
        #'threeyearly_prior_scale': [0.005, 0.5, 20, 40],
        #'client_prior' : [0.001, 1, 10, 40],
        #'close_prior' : [0.001, 1, 10, 30],
        #'price_am_prior' : [0.001, 1, 10, 30],
        #'price_pm_prior' : [0.001, 1, 10, 30],
        #'month_prior' : [0.001, 1, 10, 50],
        #'dayofyear_prior' : [0.001, 1, 10, 40],
        #'kyureki_prior' : [0.001, 1, 10, 30],
        #'busy_prior' : [0.001, 1, 10, 30],
        #'rain_prior' : [0.001, 1, 10, 30],
    }

    print('generate params')

    # Generate all combinations of parameters
    import itertools
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    mae = []  # Store the MAEs for each params here

    print('load data')
    train = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)
    train = features_create_for_y(train, train)

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
    holidays = pd.DataFrame({
        'holiday': holiday_names,
        'ds': holidays,
        'lower_window': 0,
        'upper_window': 1,
    })
    train.reset_index(inplace=True)
    train = train.rename(columns={'datetime': 'ds'})

    print('start loop')
    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(
            holidays=holidays,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale = params['changepoint_prior_scale'],
            changepoint_range = params['changepoint_range'],
            #growth = all_params['growth'],
            holidays_prior_scale = params['holidays_prior_scale'],
            n_changepoints = params['n_changepoints'],
            seasonality_mode='multiplicative'
        ).add_seasonality(
            name='threeyearly',
            period=365*3,
            fourier_order = params['threeyearly_fourier'],
            #prior_scale = all_params['threeyearly_prior_scale'],
            mode='multiplicative'
        ).add_seasonality(
            name='yearly',
            period=365,
            fourier_order = params['yearly_fourier'],
            #prior_scale = all_params['yearly_prior_scale']
        ).add_seasonality(
            name='monthly',
            period=28,
            fourier_order = params['monthly_fourier'],
            #prior_scale = all_params['monthly_prior_scale'], 
            mode='multiplicative'
        ).add_seasonality(
            name='weekly',
            period=7,
            fourier_order = params['weekly_fourier'],
            #prior_scale = all_params['weekly_prior_scale'],
            mode='multiplicative'
        ).add_seasonality(
            name='quaterly',
            period=365/4,
            fourier_order = params['quaterly_fourier'],
            #prior_scale = all_params['quaterly_prior_scale'],
            mode='multiplicative'
        ).add_regressor(
            name='client',
            #prior_scale = all_params['client_prior'],
        ).add_regressor(
            name='close',
            #prior_scale = all_params['close_prior'],
        ).add_regressor(
            name='price_am',
            #prior_scale = all_params['price_am_prior'],
        ).add_regressor(
            name='price_pm',
            #prior_scale = all_params['price_pm_prior'],
        ).add_regressor(
            name='Month',
            #prior_scale = all_params['month_prior'],
        ).add_regressor(
            name='DayOfYear',
            #prior_scale = all_params['dayofyear_prior'],
        ).add_regressor(
            name='Kyureki',
            #prior_scale = all_params['kyureki_prior'],
        ).add_regressor(
            name='busy',
            #prior_scale = all_params['busy_prior'],
        ).add_regressor(
            name='rain',
            #prior_scale = all_params['rain_prior'],
        ).fit(train) 

        from prophet.diagnostics import cross_validation
        df_cv = cross_validation(m, initial='1095 days', period='30 days', horizon = '365 days', parallel="processes")
        from prophet.diagnostics import performance_metrics
        df_p = performance_metrics(df_cv, rolling_window=1)
        mae.append(df_p['mae'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['mae'] = mae
    tuning_results.to_csv('tuning_results.csv', index=False, header=True)

def inference_with_params():
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
    holidays = pd.DataFrame({
        'holiday': holiday_names,
        'ds': holidays,
        'lower_window': 0,
        'upper_window': 1,
    })

    # seasonality
    from prophet import Prophet

    # seasonality
    
    m=Prophet(
        holidays=holidays,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        #growth = 'logistic',
        holidays_prior_scale=3.9,
        changepoint_range=0.378,
        n_changepoints=23,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=30.5, 
        seasonality_mode='multiplicative'
    )
    m.add_seasonality(name='threeyearly', period=365*3, fourier_order=7, prior_scale=18.501, mode='multiplicative')
    m.add_seasonality(name='yearly', period=365, fourier_order=21,  prior_scale=13.501)
    m.add_seasonality(name='monthly', period=28, fourier_order=1,  prior_scale=4.001, mode='multiplicative')
    m.add_seasonality(name='weekly', period=7, fourier_order=1,  prior_scale=26.001, mode='multiplicative')
    m.add_seasonality(name='quaterly', period=365/4, fourier_order=5,  prior_scale=12.501, mode='multiplicative')
    m.add_regressor(name='client', prior_scale=40.0)
    m.add_regressor(name='close', prior_scale=25.0)
    m.add_regressor(name='price_am', prior_scale=1.0)
    m.add_regressor(name='price_pm', prior_scale=22.00)
    m.add_regressor(name='Month', prior_scale=58)
    m.add_regressor(name='DayOfYear', prior_scale=12)
    m.add_regressor(name='Kyureki', prior_scale=29.5)
    m.add_regressor(name='busy', prior_scale=1.0)
    m.add_regressor(name='rain', prior_scale=2.0)
    """
    m=Prophet(
        holidays=holidays,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False, 
        #growth = 'logistic',
        seasonality_mode='multiplicative'
    )
    m.add_seasonality(name='threeyearly', period=365*3, fourier_order=22, mode='multiplicative')
    m.add_seasonality(name='yearly', period=365, fourier_order=22)
    m.add_seasonality(name='monthly', period=28, fourier_order=4, mode='multiplicative')
    m.add_seasonality(name='weekly', period=7, fourier_order=2, mode='multiplicative')
    m.add_seasonality(name='quaterly', period=365/4, fourier_order=12, mode='multiplicative')
    m.add_regressor(name='client')
    m.add_regressor(name='close')
    m.add_regressor(name='price_am')
    m.add_regressor(name='price_pm')
    m.add_regressor(name='Month')
    m.add_regressor(name='DayOfYear')
    m.add_regressor(name='Kyureki')
    m.add_regressor(name='busy')
    m.add_regressor(name='rain')
    """

    train.to_csv('train_new.csv', index=True, header=True)
    test.to_csv('test_new.csv', index=True, header=True)
    
    train = pd.read_csv("./train_new.csv", low_memory=False, parse_dates=[0])
    test = pd.read_csv("./test_new.csv", low_memory=False, parse_dates=[0])
    train = train.rename(columns={'datetime': 'ds'})
    test = test.rename(columns={'datetime': 'ds'})

   
    m.fit(train)      
    preds_y = m.predict(test)
    submission = pd.DataFrame(preds_y['ds'])
    submission['y'] = preds_y['yhat']
    submission.to_csv('propher_3year_fit.csv', index=False, header=False)

if __name__ == "__main__":
    #check_params()
    tune_params()
    #tune_params_grid()
    #inference_with_params()