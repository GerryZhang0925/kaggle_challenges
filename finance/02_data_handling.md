# Data Handling

## 1. Slicing and concatenation
```
first_array = np.array([ 1, 2, 3, 5, 8, 13])
second_array = np.array([21, 34, 55, 89, 144, 233])
# Reshaping the arrays so they become dimensionally compatible
first_array = np.reshape(first_array, (–1, 1))
second_array = np.reshape(second_array, (–1, 1))
# Concatenating both arrays by columns
combined_array = np.concatenate((first_array, second_array), axis = 1)
# Concatenating both arrays by rows
combined_array = np.concatenate((first_array, second_array), axis = 0)
```
```
first_data_frame = pd.DataFrame({'first_column' : [ 1, 2, 3],
'second_column' : [ 4, 5, 6]})
second_data_frame = pd.DataFrame({'first_column' : [ 7, 8, 9],
'second_column' : [10, 11, 12]})
# Concatenating both dataframes by columns
combined_data_frame = pd.concat([first_data_frame, second_data_frame],
axis = 1)
# Concatenating both dataframes by rows
combined_data_frame = pd.concat([first_data_frame, second_data_frame],
axis = 0)
```

## 2. Start Trades
```
pip install MetaTrader5
```
```
import datetime # Gives tools for manipulating dates and time
import pytz # Offers cross-platform time zone calculations
import MetaTrader5 as mt5 # Importing the software's library
import pandas as pd
import numpy as np

frame_M15 = mt5.TIMEFRAME_M15 # 15-minute time frame
frameframe_M30 = mt5.TIMEFRAME_M30 # 30-minute time frame
frame_H1 = mt5.TIMEFRAME_H1 # Hourly time frame
frame_H4 = mt5.TIMEFRAME_H4 # 4-hour time frame
frame_D1 = mt5.TIMEFRAME_D1 # Daily time frame
frame_W1 = mt5.TIMEFRAME_W1 # Weekly time frame
frame_M1 = mt5.TIMEFRAME_MN1 # Monthly time frame

now = datetime.datetime.now()
assets = ['EURUSD', 'USDCHF', 'GBPUSD', 'USDCAD']

def get_quotes(time_frame, year = 2005, month = 1, day = 1,
asset = "EURUSD"):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    timezone = pytz.timezone("Europe/Paris")
    time_from = datetime.datetime(year, month, day, tzinfo = timezone)
    time_to = datetime.datetime.now(timezone) + datetime.timedelta(days=1)
    rates = mt5.copy_rates_range(asset, time_frame, time_from, time_to)
    rates_frame = pd.DataFrame(rates)
    return rates_frame
```

Following timezone can be used.
```
America/New_York
Europe/London
Europe/Paris
Asia/Tokyo
Australia/Sydney
```