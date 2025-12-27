# url
`
[Apple Moving](https://signate.jp/competitions/269)

# set up environment
```
conda create -y --force -n ag python=3.8 pip
conda activate ag
pip install "mxnet<2.0.0"
pip install autogluon
pip install jpholiday
pip install xgboost
pip install seaborn
```
# history
- run_automl.py : try TimeSeriesPredictor of autogluon
- run_lgbm.py : try the solution from the network without solving the missing data problem
- EDA.ipynb : try EDA for the problem
- EDA2.ipynb : try to solve the missing data problem 
- run_xgb.py : solution of the missing data problem which focuses on all the missing data
- run_2step.py : 2 step solution focusing on a part of the missing data, the first step predicts price_am and price_pm, the second step predicts y
- run_autotime.py : combination of time series prediction and tree-based prediction