# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from time import time
import jpholiday
import datetime as dt

def create_existing_data(train_src, test_src):
    train_df = train_src.copy()
    train_df = train_df.drop('y', axis='columns')
    test_df = test_src.copy()
    
    data = pd.concat([train_df, test_df])
    data = data[data.price_am != -1]
    data = data[data.price_pm != -1]
    data.set_index('datetime', inplace=True)
    return data
    
def create_non_existing_data(train_src, test_src):
    train_df = train_src.copy()
    train_df = train_df.drop('y', axis='columns')
    test_df = test_src.copy()
    
    data = pd.concat([train_df, test_df])
    data = data[(data.price_am == -1) | (data.price_pm == -1)]
    data.set_index('datetime', inplace=True)
    return data

def get_mean_am_of_same_day(existing_data, target_data):
    price_am_list = []
    price_pm_list = []
    price_am_list_prev = []
    price_pm_list_prev = []
    price_am_list_post = []
    price_pm_list_post = []
    for i in target_data.index:
        price_am = []
        price_pm = []
        price_am_prev = []
        price_pm_prev = []
        price_am_post = []
        price_pm_post = []
        prev_day = i + pd.DateOffset(days=-1)
        post_day = i + pd.DateOffset(days=1)
        
        for y in np.arange(2010, 2017):
            # for current day
            year = i.year
            month = i.month
            day = i.day
            if month == 2 and day == 29:
                day = 28
            if y != year and dt.datetime(y, month, day) in existing_data.index:
                price_am.append(existing_data.loc[dt.datetime(y, month, day), 'price_am'])
                price_pm.append(existing_data.loc[dt.datetime(y, month, day), 'price_pm'])
                
        for y in np.arange(2010, 2017):
            # for previous day
            year = prev_day.year
            month = prev_day.month
            day = prev_day.day
            if month == 2 and day == 29:
                day = 28
            if y != year and dt.datetime(y, month, day) in existing_data.index:
                price_am_prev.append(existing_data.loc[dt.datetime(y, month, day), 'price_am'])
                price_pm_prev.append(existing_data.loc[dt.datetime(y, month, day), 'price_pm'])
                
        for y in np.arange(2010, 2017):
            # for post day
            year = post_day.year
            month = post_day.month
            day = post_day.day
            if month == 2 and day == 29:
                day = 28
            if y != year and dt.datetime(y, month, day) in existing_data.index:
                price_am_post.append(existing_data.loc[dt.datetime(y, month, day), 'price_am'])
                price_pm_post.append(existing_data.loc[dt.datetime(y, month, day), 'price_pm'])
        
        if len(price_am) == 0:
            price_am_list.append(0)
        else:
            price_am_list.append(np.mean(price_am))
            
        if len(price_pm) == 0:
            price_pm_list.append(0)
        else:
            price_pm_list.append(np.mean(price_pm))
        
        if len(price_am_prev) == 0:
            price_am_list_prev.append(0)
        else:
            price_am_list_prev.append(np.mean(price_am_prev))
                  
        if len(price_pm_prev) == 0:
            price_pm_list_prev.append(0)
        else:
            price_pm_list_prev.append(np.mean(price_pm_prev))
                  
        if len(price_am_post) == 0:
            price_am_list_post.append(0)
        else:
            price_am_list_post.append(np.mean(price_am_post))
            
        if len(price_pm_post) == 0:
            price_pm_list_post.append(0)
        else:
            price_pm_list_post.append(np.mean(price_pm_post))
            
    return price_am_list, price_pm_list, price_am_list_prev, price_pm_list_prev, price_am_list_post, price_pm_list_post
    
def create_dataset_price_train(train_src, test_src):
    holiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    holiday = [x[0] for x in holiday]
    
    base_data = create_existing_data(train_src, test_src)
    base_data['Year'] = base_data.index.year
    base_data['Month'] = base_data.index.month
    base_data['Day'] = base_data.index.day
    base_data['DayOfWeek'] = base_data.index.dayofweek
    base_data['WeekOfYear'] = base_data.index.isocalendar().week
    base_data['JpHoliday'] = [1 if d.date() in holiday else 0 for d in base_data.index]
    #price_am_list, price_pm_list, price_am_list_prev, price_pm_list_prev, price_am_list_post, price_pm_list_post = get_mean_am_of_same_day(base_data, base_data)
    #base_data['MeanPriceAMOfSameDayInOtherYear'] = price_am_list
    #base_data['MeanPricePMOfSameDayInOtherYear'] = price_pm_list
    #base_data['MeanPriceAMOfPreviousDayInOtherYear'] = price_am_list_prev
    #base_data['MeanPricePMOfPreviousDayInOtherYear'] = price_pm_list_prev
    #base_data['MeanPriceAMOfPostDayInOtherYear'] = price_am_list_post
    #base_data['MeanPricePMOfPostDayInOtherYear'] = price_pm_list_post
    
    return base_data

def create_dataset_price_test(train_src, test_src):
    holiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    holiday = [x[0] for x in holiday]
    
    lib_data = create_existing_data(train_src, test_src)
    base_data = create_non_existing_data(train_src, test_src)
    base_data['Year'] = base_data.index.year
    base_data['Month'] = base_data.index.month
    base_data['Day'] = base_data.index.day
    base_data['DayOfWeek'] = base_data.index.dayofweek
    base_data['WeekOfYear'] = base_data.index.isocalendar().week
    base_data['JpHoliday'] = [1 if d.date() in holiday else 0 for d in base_data.index]
    #price_am_list, price_pm_list, price_am_list_prev, price_pm_list_prev, price_am_list_post, price_pm_list_post = get_mean_am_of_same_day(lib_data, base_data)
    #base_data['MeanPriceAMOfSameDayInOtherYear'] = price_am_list
    #base_data['MeanPricePMOfSameDayInOtherYear'] = price_pm_list
    #base_data['MeanPriceAMOfPreviousDayInOtherYear'] = price_am_list_prev
    #base_data['MeanPricePMOfPreviousDayInOtherYear'] = price_pm_list_prev
    #base_data['MeanPriceAMOfPostDayInOtherYear'] = price_am_list_post
    #base_data['MeanPricePMOfPostDayInOtherYear'] = price_pm_list_post
    X_test_price = base_data.drop(['price_am', 'price_pm'], axis='columns')
    
    return X_test_price

def fix_missing_value(target_data, preds_price_am, preds_price_pm):
    data = target_data.copy()
    for index, row in data.iterrows():
        if row['price_am'] == -1:
            data.at[index, 'price_am'] = preds_price_am[row['datetime']]
        if row['price_pm'] == -1:
            data.at[index, 'price_pm'] = preds_price_pm[row['datetime']]
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

def features_create(src):  
    holiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    holiday = [x[0] for x in holiday]
    data = src.copy()

    # normal features
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['DayOfWeek'] = data.index.dayofweek
    data['WeekOfYear'] = data.index.isocalendar().week
    data['JpHoliday'] = [1 if d.date() in holiday else 0 for d in data.index]

    #data['oneday_before_am'] = data['price_am'].shift(1)
    #data['oneday_before_pm'] = data['price_pm'].shift(1)
    #data['rel_3am'] = data['price_am'].rolling(7).sum()
    #data['rel_3pm'] = data['price_pm'].rolling(7).sum()
    #data['rel_3cli'] = data['client'].rolling(7).sum()
    #data['rel_3clo'] = data['close'].rolling(7).sum()

    # features for time series
    data.sort_values(['datetime'],ascending = True, inplace=True)
    
    return data


if __name__ == "__main__":
    # read data
    train = pd.read_csv("./train.csv",parse_dates=[0], low_memory=False)
    test = pd.read_csv("./test.csv",parse_dates=[0], low_memory=False)
    train = train.astype({"y": float})

    # predict price_am and price_pm
    print('*****STEP1***** Predict price_am')
    X_train_price = create_dataset_price_train(train, test)
    X_test_price = create_dataset_price_test(train, test)

    import torch
    torch.set_float32_matmul_precision('high')

    from autogluon.tabular import TabularDataset, TabularPredictor
    predictor_price_am = TabularPredictor(label='price_am').fit(
        X_train_price.drop(columns=['price_pm']),
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )
    print('*****STEP2***** Predict price_pm')
    predictor_price_pm = TabularPredictor(label='price_pm').fit(
        X_train_price.drop(columns=['price_am']),
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )
    preds_price_am = predictor_price_am.predict(X_test_price)
    preds_price_pm = predictor_price_pm.predict(X_test_price)
    fixed_train = fix_missing_value(train, preds_price_am, preds_price_pm)
    fixed_test = fix_missing_value(test, preds_price_am, preds_price_pm)

    # predict y
    print('*****STEP3***** Predict y')
    X_train_y = features_create(fixed_train)
    X_test_y = features_create(fixed_test)

    predictor_y = TabularPredictor(label='y').fit(
        X_train_y,
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )
    preds_y = predictor_y.predict(X_test_y)
    preds_y.to_csv('submissionLL.csv', index=True, header=False)