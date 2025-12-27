import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error as MAE

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#上限表示数を拡張
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
sample = pd.read_csv('./sample_submit.csv',header=None)

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
sample = reduce_mem_usage(sample)

#四則演算
train['am*pm'] = train['price_am']*train['price_pm']
train['cli*clo'] = train['client']*train['close']
train['cli*am'] = train['client']*train['price_am']
train['cli*pm'] = train['client']*train['price_pm']
train['clo*am'] = train['close']*train['price_am']
train['clo*pm'] = train['close']*train['price_pm']

train['am+pm'] = train['price_am']+train['price_pm']
train['cli+clo'] = train['client']+train['close']
train['cli+am'] = train['client']+train['price_am']
train['cli+pm'] = train['client']+train['price_pm']
train['clo+am'] = train['close']+train['price_am']
train['clo+pm'] = train['close']+train['price_pm']

train['am-pm'] = train['price_am']-train['price_pm']
train['cli-clo'] = train['client']-train['close']
train['cli-am'] = train['client']-train['price_am']
train['cli-pm'] = train['client']-train['price_pm']
train['clo-am'] = train['close']-train['price_am']
train['clo-pm'] = train['close']-train['price_pm']

#ラグ特徴量、移動平均、日付
train['oneday_before_am'] = train['price_am'].shift(1)
train['oneday_before_pm'] = train['price_pm'].shift(1)
train['rel_3am'] = train['price_am'].rolling(7).sum()
train['rel_3pm'] = train['price_pm'].rolling(7).sum()
train['rel_3cli'] = train['client'].rolling(7).sum()
train['rel_3clo'] = train['close'].rolling(7).sum()
train['datetime'] = pd.to_datetime(train['datetime']) 
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['dayofweek'] = train['datetime'].dt.dayofweek

test['am*pm'] = test['price_am']*test['price_pm']
test['cli*clo'] = test['client']*test['close']
test['cli*am'] = test['client']*test['price_am']
test['cli*pm'] = test['client']*test['price_pm']
test['clo*am'] = test['close']*test['price_am']
test['clo*pm'] = test['close']*test['price_pm']

test['am+pm'] = test['price_am']+test['price_pm']
test['cli+clo'] = test['client']+test['close']
test['cli+am'] = test['client']+test['price_am']
test['cli+pm'] = test['client']+test['price_pm']
test['clo+am'] = test['close']+test['price_am']
test['clo+pm'] = test['close']+test['price_pm']

test['am-pm'] = test['price_am']-test['price_pm']
test['cli-clo'] = test['client']-test['close']
test['cli-am'] = test['client']-test['price_am']
test['cli-pm'] = test['client']-test['price_pm']
test['clo-am'] = test['close']-test['price_am']
test['clo-pm'] = test['close']-test['price_pm']

test['oneday_before_am'] = test['price_am'].shift(1)
test['oneday_before_pm'] = test['price_pm'].shift(1)
test['rel_3am'] = test['price_am'].rolling(7).sum()
test['rel_3pm'] = test['price_pm'].rolling(7).sum()
test['rel_3cli'] = test['client'].rolling(7).sum()
test['rel_3clo'] = test['close'].rolling(7).sum()
test['datetime'] = pd.to_datetime(test['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['dayofweek'] = test['datetime'].dt.dayofweek

# Run prediction
X_train = train.drop(['y','datetime'], axis=1)
y_train = train['y']
X_test = test.drop(['datetime'], axis=1)
model = lgb.LGBMRegressor(random_state=42,n_estimators=500)
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
MAE(y_train,y_pred_train)

y_pred_test = model.predict(X_test)

sample[1] = np.array(y_pred_test) + 1
#日付の区切りを'/'から'-'へ
sample[0] = pd.to_datetime(sample[0])
sample.to_csv('submit_lgbm.csv', header=False, index=False)