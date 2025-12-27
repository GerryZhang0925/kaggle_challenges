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

def fix_dataset_for_price(train):
    data = train.copy()
    for i in data.index:
        if i.date() > dt.date(2011, 1, 4):
            if (data.loc[i, 'close'] == 0) and (data.loc[i, 'price_pm'] == -1):
                data.loc[i, 'price_pm'] = 0
                data.loc[i, 'price_am'] = 0
    return data

def slip_train_val_for_price(train, start_date, offset):
    end_date = start_date + dt.timedelta(days=offset-1)
    return train[start_date:(end_date + dt.timedelta(days=-1))], train[end_date:]

def features_add_current_for_price(price_data, src):
    data = src.copy()

    price_w_mean = price_data.resample('W').mean()
    price_w_min = price_data.resample('W').min()
    price_w_max = price_data.resample('W').max()
    price_w_std = price_data.resample('W').std()
    price_w_sum = price_data.resample('W').sum()
    price_w_skew = price_data.resample('W').agg(skew)
    price_w_kurt = price_data.resample('W').agg(kurtosis)

    price_m_mean = price_data.resample('M').mean()
    price_m_min = price_data.resample('M').min()
    price_m_max = price_data.resample('M').max()
    price_m_std = price_data.resample('M').std()
    price_m_sum = price_data.resample('M').sum()
    price_m_skew = price_data.resample('M').agg(skew)
    price_m_kurt = price_data.resample('M').agg(kurtosis)

    price_q_mean = price_data.resample('Q').mean()
    price_q_min = price_data.resample('Q').min()
    price_q_max = price_data.resample('Q').max()
    price_q_std = price_data.resample('Q').std()
    price_q_sum = price_data.resample('Q').sum()
    price_q_skew = price_data.resample('Q').agg(skew)
    price_q_kurt = price_data.resample('Q').agg(kurtosis)

    for i in data.index:
        # week information
        for j in price_w_mean.index:
            resample_year = j.year
            resample_week = j.week
            target_date = i + relativedelta(years=1) 
            target_year = target_date.year
            target_week = target_date.week
            #pre_target_date = target_date + relativedelta(weeks=-1)
            pre_target_date = target_date + relativedelta(months=-3)
            pre_target_year = pre_target_date.year
            pre_target_week = pre_target_date.week
            prepre_target_date = target_date + relativedelta(months=-6)
            prepre_target_year = prepre_target_date.year
            prepre_target_week = prepre_target_date.week
            preprepre_target_date = target_date + relativedelta(months=-9)
            preprepre_target_year = preprepre_target_date.year
            preprepre_target_week = preprepre_target_date.week
            # week of next year
            if resample_year == target_year and resample_week == target_week:
                data.loc[i, 'PriceAmWeekMeanNextYear'] = price_w_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekMinNextYear'] = price_w_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekMaxNextYear'] = price_w_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekStdNextYear'] = price_w_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekSumNextYear'] = price_w_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekSkewNextYear'] = price_w_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmWeekMeanNextYear'] = price_w_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekMinNextYear'] = price_w_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekMaxNextYear'] = price_w_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekStdNextYear'] = price_w_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekSumNextYear'] = price_w_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekSkewNextYear'] = price_w_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_pm']
            # pre_week of next year
            """
            elif resample_year == pre_target_year and resample_week == pre_target_week:
                data.loc[i, 'PriceAmPreWeekMeanNextYear'] = price_w_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreWeekMinNextYear'] = price_w_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreWeekMaxNextYear'] = price_w_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreWeekStdNextYear'] = price_w_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreWeekSumNextYear'] = price_w_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreWeekSkewNextYear'] = price_w_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPreWeekMeanNextYear'] = price_w_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreWeekMinNextYear'] = price_w_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreWeekMaxNextYear'] = price_w_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreWeekStdNextYear'] = price_w_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreWeekSumNextYear'] = price_w_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreWeekSkewNextYear'] = price_w_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_pm']
            # prepre_week of next year
            elif resample_year == prepre_target_year and resample_week == prepre_target_week:
                data.loc[i, 'PriceAmPrePreWeekMeanNextYear'] = price_w_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreWeekMinNextYear'] = price_w_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreWeekMaxNextYear'] = price_w_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreWeekStdNextYear'] = price_w_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreWeekSumNextYear'] = price_w_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreWeekSkewNextYear'] = price_w_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPrePreWeekMeanNextYear'] = price_w_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreWeekMinNextYear'] = price_w_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreWeekMaxNextYear'] = price_w_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreWeekStdNextYear'] = price_w_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreWeekSumNextYear'] = price_w_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreWeekSkewNextYear'] = price_w_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_pm']
            # preprepre_week of next year
            elif resample_year == preprepre_target_year and resample_week == preprepre_target_week:
                data.loc[i, 'PriceAmPrePrePreWeekMeanNextYear'] = price_w_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreWeekMinNextYear'] = price_w_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreWeekMaxNextYear'] = price_w_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreWeekStdNextYear'] = price_w_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreWeekSumNextYear'] = price_w_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreWeekSkewNextYear'] = price_w_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPrePrePreWeekMeanNextYear'] = price_w_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreWeekMinNextYear'] = price_w_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreWeekMaxNextYear'] = price_w_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreWeekStdNextYear'] = price_w_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreWeekSumNextYear'] = price_w_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreWeekSkewNextYear'] = price_w_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreWeekKurtNextYear'] = price_w_kurt.loc[j, 'price_pm']
            """
        # month information
        for j in price_m_mean.index:
            resample_year = j.year
            resample_month = j.month
            target_date = i + relativedelta(years=1) 
            target_year = target_date.year
            target_month = target_date.month
            #pre_target_date = target_date + relativedelta(months=-1)
            pre_target_date = target_date + relativedelta(months=-3)
            pre_target_year = pre_target_date.year
            pre_target_month = pre_target_date.month
            prepre_target_date = target_date + relativedelta(months=-6)
            prepre_target_year = prepre_target_date.year
            prepre_target_month = prepre_target_date.month
            preprepre_target_date = target_date + relativedelta(months=-9)
            preprepre_target_year = preprepre_target_date.year
            preprepre_target_month = preprepre_target_date.month
            # month of next year
            if resample_year == target_year and resample_month == target_month:
                data.loc[i, 'PriceAmMonthMeanNextYear'] = price_m_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthMinNextYear'] = price_m_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthMaxNextYear'] = price_m_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthStdNextYear'] = price_m_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthSumNextYear'] = price_m_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthSkewNextYear'] = price_m_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmMonthMeanNextYear'] = price_m_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthMinNextYear'] = price_m_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthMaxNextYear'] = price_m_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthStdNextYear'] = price_m_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthSumNextYear'] = price_m_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthSkewNextYear'] = price_m_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_pm']
            # pre_month of next year
            """
            elif resample_year == pre_target_year and resample_month == pre_target_month:
                data.loc[i, 'PriceAmPreMonthMeanNextYear'] = price_m_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreMonthMinNextYear'] = price_m_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreMonthMaxNextYear'] = price_m_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreMonthStdNextYear'] = price_m_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreMonthSumNextYear'] = price_m_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreMonthSkewNextYear'] = price_m_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPreMonthMeanNextYear'] = price_m_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreMonthMinNextYear'] = price_m_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreMonthMaxNextYear'] = price_m_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreMonthStdNextYear'] = price_m_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreMonthSumNextYear'] = price_m_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreMonthSkewNextYear'] = price_m_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_pm']
            # prepre_month of next year
            elif resample_year == prepre_target_year and resample_month == prepre_target_month:
                data.loc[i, 'PriceAmPrePreMonthMeanNextYear'] = price_m_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreMonthMinNextYear'] = price_m_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreMonthMaxNextYear'] = price_m_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreMonthStdNextYear'] = price_m_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreMonthSumNextYear'] = price_m_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreMonthSkewNextYear'] = price_m_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPrePreMonthMeanNextYear'] = price_m_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreMonthMinNextYear'] = price_m_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreMonthMaxNextYear'] = price_m_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreMonthStdNextYear'] = price_m_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreMonthSumNextYear'] = price_m_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreMonthSkewNextYear'] = price_m_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_pm']
            # preprepre_month of next year
            elif resample_year == preprepre_target_year and resample_month == preprepre_target_month:
                data.loc[i, 'PriceAmPrePrePreMonthMeanNextYear'] = price_m_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreMonthMinNextYear'] = price_m_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreMonthMaxNextYear'] = price_m_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreMonthStdNextYear'] = price_m_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreMonthSumNextYear'] = price_m_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreMonthSkewNextYear'] = price_m_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPrePrePreMonthMeanNextYear'] = price_m_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreMonthMinNextYear'] = price_m_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreMonthMaxNextYear'] = price_m_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreMonthStdNextYear'] = price_m_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreMonthSumNextYear'] = price_m_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreMonthSkewNextYear'] = price_m_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreMonthKurtNextYear'] = price_m_kurt.loc[j, 'price_pm']
            """
        # quarter information
        for j in price_q_mean.index:
            resample_year = j.year
            resample_quarter = j.quarter
            target_date = i + relativedelta(years=1) 
            target_year = target_date.year
            target_quarter = target_date.quarter
            pre_target_date = target_date + relativedelta(months=-3)
            pre_target_year = pre_target_date.year
            pre_target_quarter = pre_target_date.quarter
            prepre_target_date = target_date + relativedelta(months=-6)
            prepre_target_year = prepre_target_date.year
            prepre_target_quarter = prepre_target_date.quarter
            preprepre_target_date = target_date + relativedelta(months=-9)
            preprepre_target_year = preprepre_target_date.year
            preprepre_target_quarter = preprepre_target_date.quarter
            # quarter of next year
            if resample_year == target_year and resample_quarter == target_quarter:
                data.loc[i, 'PriceAmQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterMinNextYear'] = price_q_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterMaxNextYear'] = price_q_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterStdNextYear'] = price_q_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterSumNextYear'] = price_q_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterMinNextYear'] = price_q_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterMaxNextYear'] = price_q_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterStdNextYear'] = price_q_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterSumNextYear'] = price_q_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_pm']
            # pre_quarter of next year
            """
            elif resample_year == pre_target_year and resample_quarter == pre_target_quarter:
                data.loc[i, 'PriceAmPreQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreQuarterMinNextYear'] = price_q_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreQuarterMaxNextYear'] = price_q_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreQuarterStdNextYear'] = price_q_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreQuarterSumNextYear'] = price_q_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPreQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPreQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreQuarterMinNextYear'] = price_q_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreQuarterMaxNextYear'] = price_q_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreQuarterStdNextYear'] = price_q_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreQuarterSumNextYear'] = price_q_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPreQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_pm']
            # prepre_quarter of next year
            elif resample_year == prepre_target_year and resample_quarter == prepre_target_quarter:
                data.loc[i, 'PriceAmPrePreQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreQuarterMinNextYear'] = price_q_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreQuarterMaxNextYear'] = price_q_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreQuarterStdNextYear'] = price_q_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreQuarterSumNextYear'] = price_q_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePreQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPrePreQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreQuarterMinNextYear'] = price_q_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreQuarterMaxNextYear'] = price_q_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreQuarterStdNextYear'] = price_q_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreQuarterSumNextYear'] = price_q_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePreQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_pm']
            # preprepre_quarter of next year
            elif resample_year == preprepre_target_year and resample_quarter == preprepre_target_quarter:
                data.loc[i, 'PriceAmPrePrePreQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreQuarterMinNextYear'] = price_q_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreQuarterMaxNextYear'] = price_q_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreQuarterStdNextYear'] = price_q_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreQuarterSumNextYear'] = price_q_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmPrePrePreQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmPrePrePreQuarterMeanNextYear'] = price_q_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreQuarterMinNextYear'] = price_q_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreQuarterMaxNextYear'] = price_q_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreQuarterStdNextYear'] = price_q_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreQuarterSumNextYear'] = price_q_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreQuarterSkewNextYear'] = price_q_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmPrePrePreQuarterKurtNextYear'] = price_q_kurt.loc[j, 'price_pm']
            """
    # Create Ratio Information
    """
    data['PriceAmWeekMeanDiff'] = (data['PriceAmWeekMeanNextYear'] - data['PriceAmPreWeekMeanNextYear'])
    data['PriceAmWeekMinDiff'] = (data['PriceAmWeekMinNextYear'] - data['PriceAmPreWeekMinNextYear'])
    data['PriceAmWeekMaxDiff'] = (data['PriceAmWeekMaxNextYear'] - data['PriceAmPreWeekMaxNextYear'])
    data['PriceAmWeekStdDiff'] = (data['PriceAmWeekStdNextYear'] - data['PriceAmPreWeekStdNextYear'])
    data['PriceAmWeekSumDiff'] = (data['PriceAmWeekSumNextYear'] - data['PriceAmPreWeekSumNextYear'])
    data['PriceAmWeekSkewDiff'] = (data['PriceAmWeekSkewNextYear'] - data['PriceAmPreWeekSkewNextYear'])
    data['PriceAmWeekKurtDiff'] = (data['PriceAmWeekKurtNextYear'] - data['PriceAmPreWeekKurtNextYear'])

    data['PricePmWeekMeanDiff'] = (data['PricePmWeekMeanNextYear'] - data['PricePmPreWeekMeanNextYear'])
    data['PricePmWeekMinDiff'] = (data['PricePmWeekMinNextYear'] - data['PricePmPreWeekMinNextYear'])
    data['PricePmWeekMaxDiff'] = (data['PricePmWeekMaxNextYear'] - data['PricePmPreWeekMaxNextYear'])
    data['PricePmWeekStdDiff'] = (data['PricePmWeekStdNextYear'] - data['PricePmPreWeekStdNextYear'])
    data['PricePmWeekSumDiff'] = (data['PricePmWeekSumNextYear'] - data['PricePmPreWeekSumNextYear'])
    data['PricePmWeekSkewDiff'] = (data['PricePmWeekSkewNextYear'] - data['PricePmPreWeekSkewNextYear'])
    data['PricePmWeekKurtDiff'] = (data['PricePmWeekKurtNextYear'] - data['PricePmPreWeekKurtNextYear'])

    data['PriceAmMonthMeanDiff'] = (data['PriceAmMonthMeanNextYear'] - data['PriceAmPreMonthMeanNextYear'])
    data['PriceAmMonthMinDiff'] = (data['PriceAmMonthMinNextYear'] - data['PriceAmPreMonthMinNextYear'])
    data['PriceAmMonthMaxDiff'] = (data['PriceAmMonthMaxNextYear'] - data['PriceAmPreMonthMaxNextYear'])
    data['PriceAmMonthStdDiff'] = (data['PriceAmMonthStdNextYear'] - data['PriceAmPreMonthStdNextYear'])
    data['PriceAmMonthSumDiff'] = (data['PriceAmMonthSumNextYear'] - data['PriceAmPreMonthSumNextYear'])
    data['PriceAmMonthSkewDiff'] = (data['PriceAmMonthSkewNextYear'] - data['PriceAmPreMonthSkewNextYear'])
    data['PriceAmMonthKurtDiff'] = (data['PriceAmMonthKurtNextYear'] - data['PriceAmPreMonthKurtNextYear'])

    data['PricePmMonthMeanDiff'] = (data['PricePmMonthMeanNextYear'] - data['PricePmPreMonthMeanNextYear'])
    data['PricePmMonthMinDiff'] = (data['PricePmMonthMinNextYear'] - data['PricePmPreMonthMinNextYear'])
    data['PricePmMonthMaxDiff'] = (data['PricePmMonthMaxNextYear'] - data['PricePmPreMonthMaxNextYear'])
    data['PricePmMonthStdDiff'] = (data['PricePmMonthStdNextYear'] - data['PricePmPreMonthStdNextYear'])
    data['PricePmMonthSumDiff'] = (data['PricePmMonthSumNextYear'] - data['PricePmPreMonthSumNextYear'])
    data['PricePmMonthSkewDiff'] = (data['PricePmMonthSkewNextYear'] - data['PricePmPreMonthSkewNextYear'])
    data['PricePmMonthKurtDiff'] = (data['PricePmMonthKurtNextYear'] - data['PricePmPreMonthKurtNextYear'])

    data['PriceAmQuarterMeanDiff'] = (data['PriceAmQuarterMeanNextYear'] - data['PriceAmPreQuarterMeanNextYear'])
    data['PriceAmQuarterMinDiff'] = (data['PriceAmQuarterMinNextYear'] - data['PriceAmPreQuarterMinNextYear'])
    data['PriceAmQuarterMaxDiff'] = (data['PriceAmQuarterMaxNextYear'] - data['PriceAmPreQuarterMaxNextYear'])
    data['PriceAmQuarterStdDiff'] = (data['PriceAmQuarterStdNextYear'] - data['PriceAmPreQuarterStdNextYear'])
    data['PriceAmQuarterSumDiff'] = (data['PriceAmQuarterSumNextYear'] - data['PriceAmPreQuarterSumNextYear'])
    data['PriceAmQuarterSkewDiff'] = (data['PriceAmQuarterSkewNextYear'] - data['PriceAmPreQuarterSkewNextYear'])
    data['PriceAmQuarterKurtDiff'] = (data['PriceAmQuarterKurtNextYear'] - data['PriceAmPreQuarterKurtNextYear'])

    data['PricePmQuarterMeanDiff'] = (data['PricePmQuarterMeanNextYear'] - data['PricePmPreQuarterMeanNextYear'])
    data['PricePmQuarterMinDiff'] = (data['PricePmQuarterMinNextYear'] - data['PricePmPreQuarterMinNextYear'])
    data['PricePmQuarterMaxDiff'] = (data['PricePmQuarterMaxNextYear'] - data['PricePmPreQuarterMaxNextYear'])
    data['PricePmQuarterStdDiff'] = (data['PricePmQuarterStdNextYear'] - data['PricePmPreQuarterStdNextYear'])
    data['PricePmQuarterSumDiff'] = (data['PricePmQuarterSumNextYear'] - data['PricePmPreQuarterSumNextYear'])
    data['PricePmQuarterSkewDiff'] = (data['PricePmQuarterSkewNextYear'] - data['PricePmPreQuarterSkewNextYear'])
    data['PricePmQuarterKurtDiff'] = (data['PricePmQuarterKurtNextYear'] - data['PricePmPreQuarterKurtNextYear'])
    """
    #moving_info = pd.read_csv('moving_info.csv', low_memory=False, index_col='datetime', parse_dates=True)
    #for i in data.index:
    #    for j in moving_info.index:
    #        if i.year == j.year and i.month == j.month:
    #            data.loc[i, 'saitama'] = np.log(moving_info.loc[j, 'saitama'])
                #data.loc[i, 'chiba'] = np.log(moving_info.loc[j, 'chiba'])
                #data.loc[i, 'tokyo'] = np.log(moving_info.loc[j, 'tokyo'])
                #data.loc[i, 'kanagawa'] = np.log(moving_info.loc[j, 'kanagawa'])

    for i in data.index:
        if data.loc[i, 'DayOfWeek'] == 5 or data.loc[i, 'DayOfWeek'] == 6:
            data.loc[i, 'JpHoliday'] = 1

    # Drop Information of the current year
#    data.drop(columns=[
#                'PriceAmWeekMean', 'PriceAmWeekStd', 'PriceAmWeekSum', 'PriceAmWeekSkew', 'PriceAmWeekKurt', 'PriceAmWeekMin', 'PriceAmWeekMax',
#                'PricePmWeekMean', 'PricePmWeekStd', 'PricePmWeekSum', 'PricePmWeekSkew', 'PricePmWeekKurt', 'PricePmWeekMin', 'PricePmWeekMax',
#                'PriceAmMonthMean', 'PriceAmMonthStd', 'PriceAmMonthSum', 'PriceAmMonthSkew', 'PriceAmMonthKurt', 'PriceAmMonthMin', 'PriceAmMonthMax',
#                'PricePmMonthMean', 'PricePmMonthStd', 'PricePmMonthSum', 'PricePmMonthSkew', 'PricePmMonthKurt', 'PricePmMonthMin', 'PricePmMonthMax', 
#                'PriceAmQuarterMean', 'PriceAmQuarterStd', 'PriceAmQuarterSum', 'PriceAmQuarterSkew', 'PriceAmQuarterKurt', 'PriceAmQuarterMin', 'PriceAmQuarterMax',
#                'PricePmQuarterMean', 'PricePmQuarterStd', 'PricePmQuarterSum', 'PricePmQuarterSkew', 'PricePmQuarterKurt', 'PricePmQuarterMin', 'PricePmQuarterMax'
#                ], 
#                inplace = True)

    return data.copy()

def features_add_current_for_y(y_data, src):
    data = src.copy()

    y_w_mean = y_data.resample('W').mean()
    y_w_min = y_data.resample('W').min()
    y_w_max = y_data.resample('W').max()
    y_w_std = y_data.resample('W').std()
    y_w_sum = y_data.resample('W').sum()
    y_w_skew = y_data.resample('W').agg(skew)
    y_w_kurt = y_data.resample('W').agg(kurtosis)

    y_m_mean = y_data.resample('M').mean()
    y_m_min = y_data.resample('M').min()
    y_m_max = y_data.resample('M').max()
    y_m_std = y_data.resample('M').std()
    y_m_sum = y_data.resample('M').sum()
    y_m_skew = y_data.resample('M').agg(skew)
    y_m_kurt = y_data.resample('M').agg(kurtosis)

    y_q_mean = y_data.resample('Q').mean()
    y_q_min = y_data.resample('Q').min()
    y_q_max = y_data.resample('Q').max()
    y_q_std = y_data.resample('Q').std()
    y_q_sum = y_data.resample('Q').sum()
    y_q_skew = y_data.resample('Q').agg(skew)
    y_q_kurt = y_data.resample('Q').agg(kurtosis)
    for i in data.index:
        # week information
        for j in y_w_mean.index:
            resample_year = j.year
            resample_week = j.week
            target_date = i + relativedelta(years=-1) 
            target_year = target_date.year
            target_week = target_date.week
            delay1q_target_date = target_date + relativedelta(months=+3)
            delay1q_target_year = delay1q_target_date.year
            delay1q_target_week = delay1q_target_date.week
            # week of previous year
            if resample_year == target_year and resample_week == target_week:
                data.loc[i, 'YWeekMeanPrevYear'] = y_w_mean.loc[j, 'y']
                #data.loc[i, 'YWeekMinPrevYear'] = y_w_min.loc[j, 'y']
                #data.loc[i, 'YWeekMaxPrevYear'] = y_w_max.loc[j, 'y']
                #data.loc[i, 'YWeekStdPrevYear'] = y_w_std.loc[j, 'y']
                #data.loc[i, 'YWeekSumPrevYear'] = y_w_sum.loc[j, 'y']
                #data.loc[i, 'YWeekSkewPrevYear'] = y_w_skew.loc[j, 'y']
                #data.loc[i, 'YWeekKurtPrevYear'] = y_w_kurt.loc[j, 'y']

                """
                data.loc[i, 'PriceAmWeekMeanPrevYear'] = y_w_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekMinPrevYear'] = y_w_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekMaxPrevYear'] = y_w_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekStdPrevYear'] = y_w_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekSumPrevYear'] = y_w_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekSkewPrevYear'] = y_w_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmWeekKurtPrevYear'] = y_w_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmWeekMeanPrevYear'] = y_w_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekMinPrevYear'] = y_w_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekMaxPrevYear'] = y_w_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekStdPrevYear'] = y_w_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekSumPrevYear'] = y_w_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekSkewPrevYear'] = y_w_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmWeekKurtPrevYear'] = y_w_kurt.loc[j, 'price_pm']
                """

            # week of previous year but one quarter delay
            elif resample_year == delay1q_target_year and resample_week == delay1q_target_week:
                data.loc[i, 'YWeekMeanPrevYearDelay1Q'] = y_w_mean.loc[j, 'y']
                #data.loc[i, 'YWeekMinPrevYearDelay1Q'] = y_w_min.loc[j, 'y']
                #data.loc[i, 'YWeekMaxPrevYearDelay1Q'] = y_w_max.loc[j, 'y']
                #data.loc[i, 'YWeekStdPrevYearDelay1Q'] = y_w_std.loc[j, 'y']
                #data.loc[i, 'YWeekSumPrevYearDelay1Q'] = y_w_sum.loc[j, 'y']
                #data.loc[i, 'YWeekSkewPrevYearDelay1Q'] = y_w_skew.loc[j, 'y']
                #data.loc[i, 'YWeekKurtPrevYearDelay1Q'] = y_w_kurt.loc[j, 'y']

        # month information
        for j in y_m_mean.index:
            resample_year = j.year
            resample_month = j.month
            target_date = i + relativedelta(years=-1) 
            target_year = target_date.year
            target_month = target_date.month
            delay1q_target_date = target_date + relativedelta(months=+3)
            delay1q_target_year = delay1q_target_date.year
            delay1q_target_month = delay1q_target_date.month
            # month of previous year
            if resample_year == target_year and resample_month == target_month:
                data.loc[i, 'YMonthMeanPrevYear'] = y_m_mean.loc[j, 'y']
                #data.loc[i, 'YMonthMinPrevYear'] = y_m_min.loc[j, 'y']
                #data.loc[i, 'YMonthMaxPrevYear'] = y_m_max.loc[j, 'y']
                #data.loc[i, 'YMonthStdPrevYear'] = y_m_std.loc[j, 'y']
                #data.loc[i, 'YMonthSumPrevYear'] = y_m_sum.loc[j, 'y']
                #data.loc[i, 'YMonthSkewPrevYear'] = y_m_skew.loc[j, 'y']
                #data.loc[i, 'YMonthKurtPrevYear'] = y_m_kurt.loc[j, 'y']

                """
                data.loc[i, 'PriceAmMonthMeanPrevYear'] = y_m_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthMinPrevYear'] = y_m_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthMaxPrevYear'] = y_m_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthStdPrevYear'] = y_m_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthSumPrevYear'] = y_m_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthSkewPrevYear'] = y_m_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmMonthKurtPrevYear'] = y_m_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmMonthMeanPrevYear'] = y_m_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthMinPrevYear'] = y_m_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthMaxPrevYear'] = y_m_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthStdPrevYear'] = y_m_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthSumPrevYear'] = y_m_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthSkewPrevYear'] = y_m_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmMonthKurtPrevYear'] = y_m_kurt.loc[j, 'price_pm']
                """

            # month of previous year but one quarter delay
            elif resample_year == delay1q_target_year and resample_month == delay1q_target_month:
                data.loc[i, 'YMonthMeanPrevYearDelay1Q'] = y_m_mean.loc[j, 'y']
                #data.loc[i, 'YMonthMinPrevYearDelay1Q'] = y_m_min.loc[j, 'y']
                #data.loc[i, 'YMonthMaxPrevYearDelay1Q'] = y_m_max.loc[j, 'y']
                #data.loc[i, 'YMonthStdPrevYearDelay1Q'] = y_m_std.loc[j, 'y']
                #data.loc[i, 'YMonthSumPrevYearDelay1Q'] = y_m_sum.loc[j, 'y']
                #data.loc[i, 'YMonthSkewPrevYearDelay1Q'] = y_m_skew.loc[j, 'y']
                #data.loc[i, 'YMonthKurtPrevYearDelay1Q'] = y_m_kurt.loc[j, 'y']

        # quarter information
        for j in y_q_mean.index:
            resample_year = j.year
            resample_quarter = j.quarter
            target_date = i + relativedelta(years=-1) 
            target_year = target_date.year
            target_quarter = target_date.quarter
            delay1q_target_date = target_date + relativedelta(months=+3)
            delay1q_target_year = delay1q_target_date.year
            delay1q_target_quarter = delay1q_target_date.quarter
            # quarter of next year
            if resample_year == target_year and resample_quarter == target_quarter:
                data.loc[i, 'YQuarterMeanPrevYear'] = y_q_mean.loc[j, 'y']
                #data.loc[i, 'YQuarterMinPrevYear'] = y_q_min.loc[j, 'y']
                #data.loc[i, 'YQuarterMaxPrevYear'] = y_q_max.loc[j, 'y']
                #data.loc[i, 'YQuarterStdPrevYear'] = y_q_std.loc[j, 'y']
                #data.loc[i, 'YQuarterSumPrevYear'] = y_q_sum.loc[j, 'y']
                #data.loc[i, 'YQuarterSkewPrevYear'] = y_q_skew.loc[j, 'y']
                #data.loc[i, 'YQuarterKurtPrevYear'] = y_q_kurt.loc[j, 'y']

                """
                data.loc[i, 'PriceAmQuarterMeanPrevYear'] = y_q_mean.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterMinPrevYear'] = y_q_min.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterMaxPrevYear'] = y_q_max.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterStdPrevYear'] = y_q_std.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterSumPrevYear'] = y_q_sum.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterSkewPrevYear'] = y_q_skew.loc[j, 'price_am']
                data.loc[i, 'PriceAmQuarterKurtPrevYear'] = y_q_kurt.loc[j, 'price_am']

                data.loc[i, 'PricePmQuarterMeanPrevYear'] = y_q_mean.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterMinPrevYear'] = y_q_min.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterMaxPrevYear'] = y_q_max.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterStdPrevYear'] = y_q_std.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterSumPrevYear'] = y_q_sum.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterSkewPrevYear'] = y_q_skew.loc[j, 'price_pm']
                data.loc[i, 'PricePmQuarterKurtPrevYear'] = y_q_kurt.loc[j, 'price_pm']
                """

            # quarter of previous year but one quarter delay
            elif resample_year == delay1q_target_year and resample_month == delay1q_target_quarter:
                data.loc[i, 'YQuarterMeanPrevYearDelay1Q'] = y_q_mean.loc[j, 'y']
                #data.loc[i, 'YQuarterMinPrevYearDelay1Q'] = y_q_min.loc[j, 'y']
                #data.loc[i, 'YQuarterMaxPrevYearDelay1Q'] = y_q_max.loc[j, 'y']
                #data.loc[i, 'YQuarterStdPrevYearDelay1Q'] = y_q_std.loc[j, 'y']
                #data.loc[i, 'YQuarterSumPrevYearDelay1Q'] = y_q_sum.loc[j, 'y']
                #data.loc[i, 'YQuarterSkewPrevYearDelay1Q'] = y_q_skew.loc[j, 'y']
                #data.loc[i, 'YQuarterKurtPrevYearDelay1Q'] = y_q_kurt.loc[j, 'y']
                

    return data

def features_create_for_price(price_data, src): 
    holiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    holiday = [x[0] for x in holiday]
    data = src.copy()

    # normal features
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['DayOfWeek'] = data.index.dayofweek
    data['WeekOfYear'] = data.index.isocalendar().week
    data['DayOfYear'] = data.index.dayofyear
    data['Quarter'] = data.index.quarter
    data['JpHoliday'] = [1 if d.date() in holiday else 0 for d in data.index]

    # features for time series
    data = features_add_current_for_price(price_data, data)
    data.sort_values(['datetime'],ascending = True, inplace=True)
    return data

def create_extend_data_for_price(train_src, test_src):
    train_df = train_src.copy()
    train_df = train_df.drop('y', axis='columns')
    train_df = train_df[dt.date(2011, 1, 1):]
    test_df = test_src.copy()
    
    data = pd.concat([train_df, test_df])
    data[data['price_am'] == -1] = 0
    data[data['price_pm'] == -1] = 0 
    data.drop(columns=['client','close'], inplace=True)
    return data

def repair_train_data(train_data, price_am, price_pm):
    data = train_data.copy()
    for i in data.index:
        if i.date() < dt.date(2010, 12, 31):
            data.loc[i, 'price_am'] = price_am.loc[i, 'price_am']
            data.loc[i, 'price_pm'] = price_pm.loc[i, 'price_pm']
    return data

def find_params_for_price():
    # read data
    train = pd.read_csv("./train.csv", low_memory=False, index_col='datetime', parse_dates=True)
    test = pd.read_csv("./test.csv", low_memory=False, index_col='datetime', parse_dates=True)
    train = train.astype({"y": float})

    # predict price_am and price_pm
    print('*****STEP1***** create train/validation dataset for price_am/price_pm prediction')
    train = fix_dataset_for_price(train)
    price_val, price_train = slip_train_val_for_price(train, dt.date(2011, 1, 1), 182)
    #print(price_val.head(5))
    #print(price_val.tail(5))
    #print(price_train.head(5))
    #print(price_train.tail(5))
    #print(f"price_val: {price_val.shape}")
    #print(f"price_train: {price_train.shape}")
    price_data = create_extend_data_for_price(train, test)
    price_val = features_create_for_price(price_data, price_val)
    price_train = features_create_for_price(price_data, price_train)

    import torch
    torch.set_float32_matmul_precision('high')
    price_train = price_train.astype({"price_am": float})
    price_train = price_train.astype({"price_pm": float})

    print('*****STEP2***** train price_am')
    from autogluon.tabular import TabularDataset, TabularPredictor
    predictor_price_am = TabularPredictor(label='price_am',
                                          #problem_type='regression',
                                          #eval_metric='root_mean_squared_error'
                                          ).fit(
        price_train.drop(columns=['price_pm']),
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )
    print('*****STEP3***** train price_pm')
    predictor_price_pm = TabularPredictor(label='price_pm', 
                                          #problem_type='regression',
                                          #eval_metric='root_mean_squared_error'
                                          ).fit(
        price_train.drop(columns=['price_am']),
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )
    print('*****STEP4***** predict price')
    preds_price_am = predictor_price_am.predict(price_val.drop(columns=['price_am', 'price_pm']))
    preds_price_pm = predictor_price_pm.predict(price_val.drop(columns=['price_am', 'price_pm']))
    price_val.to_csv('price_val.csv', index=True, header=True)
    preds_price_am.to_csv('preds_price_am.csv', index=True, header=True)
    preds_price_pm.to_csv('preds_price_pm.csv', index=True, header=True)
    
    price_val = pd.read_csv("./price_val.csv",parse_dates=[0], low_memory=False)
    preds_price_am = pd.read_csv("./preds_price_am.csv",parse_dates=[0], low_memory=False)
    preds_price_pm = pd.read_csv("./preds_price_pm.csv",parse_dates=[0], low_memory=False)
    mae = metrics.mean_absolute_error(price_val['price_am'], preds_price_am['price_am'])
    rmse = np.sqrt(metrics.mean_squared_error(price_val['price_am'], preds_price_am['price_am']))
    r2 = metrics.r2_score(price_val['price_am'], preds_price_am['price_am'])
    print(f"price_am: mae({mae}), rmse({rmse}), r2({r2})")
    mae = metrics.mean_absolute_error(price_val['price_pm'], preds_price_pm['price_pm'])
    rmse = np.sqrt(metrics.mean_squared_error(price_val['price_pm'], preds_price_pm['price_pm']))
    r2 = metrics.r2_score(price_val['price_pm'], preds_price_pm['price_pm'])
    print(f"price_pm: mae({mae}), rmse({rmse}), r2({r2})")

def predict_price():
    # read data
    train = pd.read_csv("./train.csv", low_memory=False, index_col='datetime', parse_dates=True)
    test = pd.read_csv("./test.csv", low_memory=False, index_col='datetime', parse_dates=True)
    train = train.astype({"y": float})

    # predict price_am and price_pm
    print('*****STEP1***** create train/validation dataset for price_am/price_pm prediction')
    price_test, price_train = slip_train_val_for_price(train, dt.date(2010, 7, 1), 185)
    train = fix_dataset_for_price(train)
    price_data = create_extend_data_for_price(train, test)

    train = features_create_for_price(price_data, price_train)
    test = features_create_for_price(price_data, price_test)
    
    print('*****STEP2***** train price_am')
    import torch
    torch.set_float32_matmul_precision('high')
    price_train = price_train.astype({"price_am": float})
    price_train = price_train.astype({"price_pm": float})

    from autogluon.tabular import TabularDataset, TabularPredictor
    predictor_price_am = TabularPredictor(label='price_am',
                                          #problem_type='regression',
                                          #eval_metric='root_mean_squared_error'
                                          ).fit(
        train.drop(columns=['price_pm']),
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )
    print('*****STEP3***** train price_pm')
    predictor_price_pm = TabularPredictor(label='price_pm',
                                          #problem_type='regression',
                                          #eval_metric='root_mean_squared_error'
                                          ).fit(
        train.drop(columns=['price_am']),
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )
    print('*****STEP4***** predict price')
    preds_price_am = predictor_price_am.predict(test.drop(columns=['price_am', 'price_pm']))
    preds_price_pm = predictor_price_pm.predict(test.drop(columns=['price_am', 'price_pm']))
    preds_price_am.to_csv('preds_price_am.csv', index=True, header=True)
    preds_price_pm.to_csv('preds_price_pm.csv', index=True, header=True)

    print('*****STEP5***** repair price')
    train = pd.read_csv("./train.csv", low_memory=False, index_col='datetime', parse_dates=True)
    preds_price_am = pd.read_csv("./preds_price_am.csv", low_memory=False, index_col='datetime', parse_dates=True)
    preds_price_pm = pd.read_csv("./preds_price_pm.csv", low_memory=False, index_col='datetime', parse_dates=True)

    train = fix_dataset_for_price(train)
    train = repair_train_data(train, preds_price_am, preds_price_pm)
    train.to_csv('train_repaired.csv', index=True, header=True)

def features_create_for_y(y_data, src):  
    holiday = jpholiday.between(dt.date(2010, 7, 1), dt.date(2017, 3, 31))
    holiday = [x[0] for x in holiday]
    data = src.copy()

    # normal features
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['DayOfWeek'] = data.index.dayofweek
    #data['WeekOfYear'] = data.index.isocalendar().week
    data['DayOfYear'] = data.index.dayofyear
    data['Quarter'] = data.index.quarter
    #data['JpHoliday'] = [1 if d.date() in holiday else 0 for d in data.index]

    # features for time series
    #data = features_add_current_for_y(y_data, data)

    # features for time series
    data.sort_values(['datetime'],ascending = True, inplace=True)
    
    return data

def find_params_for_y():
    train = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)
    test = pd.read_csv("./test.csv", low_memory=False, index_col='datetime', parse_dates=True)
    train = train.astype({"y": float})

    # predict price_am and price_pm
    print('*****STEP1***** create train/validation dataset for y prediction')
    y_train, y_val = slip_train_val_for_price(train, dt.date(2010, 7, 1), 1736)

    print('*****STEP2***** train for y')
    train_y = features_create_for_y(y_train, y_train)
    val_y = features_create_for_y(y_train, y_val)
    #train_y = train_y[dt.date(2011, 7, 1):]

    import torch
    torch.set_float32_matmul_precision('high')

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


def predict_y():
    train = pd.read_csv("./train_repaired.csv", low_memory=False, index_col='datetime', parse_dates=True)
    test = pd.read_csv("./test.csv", low_memory=False, index_col='datetime', parse_dates=True)
    train = train.astype({"y": float})

    # predict price_am and price_pm
    print('*****STEP1***** train for y')
    train = features_create_for_y(train, train)
    test = features_create_for_y(train, test)

    import torch
    torch.set_float32_matmul_precision('high')

    from autogluon.tabular import TabularDataset, TabularPredictor
    predictor_y = TabularPredictor(label='y',
                                #problem_type='regression',
                                #eval_metric='root_mean_squared_error'
                                #eval_metric='mean_absolute_error'
                                ).fit(
        train,
        hyperparameters='multimodal',
        presets='best_quality',
        num_stack_levels=1,
        num_bag_folds=5,
        refit_full=True,
        set_best_to_refit_full=True
    )

    print('*****STEP3***** predict y')
    preds_y = predictor_y.predict(test)
    preds_y.to_csv('submission_now3.csv', index=True, header=False)
    
if __name__ == "__main__":
    #find_params_for_price()
    predict_price()
    #find_params_for_y()
    #predict_y()