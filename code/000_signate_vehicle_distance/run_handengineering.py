import torch
torch.set_float32_matmul_precision('high')

def get_first_name(carname):
    for i, c in enumerate(carname):
        if (c == ' '):
            return carname[:i]

import pandas as pd

df = pd.read_csv('train.csv')
hp_list = []
for i, h in enumerate(df['horsepower']):
    if h == '?':
        hp_list.append(int(97))
    else:
        hp_list.append(int(h))
print(hp_list)
df['horsepower'] = hp_list
        
car_list = []
for car in df['car name']:
    car_list.append(get_first_name(car))
df['short name'] = car_list
df.drop(columns=['car name'], inplace=True)

from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset(df)
id, label = 'id', 'mpg'
predictor = TabularPredictor(label=label).fit(
    train_data.drop(columns=[id]),
    hyperparameters='multimodal',
    num_stack_levels=1, num_bag_folds=5
    )

df_test = pd.read_csv('test.csv')
hp_list = []
for i, h in enumerate(df_test['horsepower']):
    if h == '?':
        hp_list.append(int(97))
    else:
        hp_list.append(int(h))
df_test['horsepower'] = hp_list
car_list = []
for car in df_test['car name']:
    car_list.append(get_first_name(car))
df_test['short name'] = car_list
df_test.drop(columns=['car name'], inplace=True)
test_data = TabularDataset(df_test)
preds = predictor.predict(test_data.drop(columns=[id]))
submission = pd.DataFrame({id:test_data[id], label:preds})
submission.to_csv('submission.csv', index=False, header=False)

