import torch
torch.set_float32_matmul_precision('high')

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

df = pd.read_csv("train.csv")
item_list = []
for i in range(df.shape[0]):
    item_list.append('H1')
df['item_id'] = item_list

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="datetime"
)
predictor = TimeSeriesPredictor(
    prediction_length=10,
    path="autogluon-daily",
    target="y",
    eval_metric="MASE",
)
predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
)

"""
test_data = TabularDataset('test.csv')
preds = predictor.predict(test_data.drop(columns=[id]))
submission = pd.DataFrame({id:test_data[id], label:preds})
submission.to_csv('submission.csv', index=False, header=False)
"""