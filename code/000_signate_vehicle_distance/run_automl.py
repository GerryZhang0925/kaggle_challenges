import torch
torch.set_float32_matmul_precision('high')

from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('train.csv')
id, label = 'id', 'mpg'
predictor = TabularPredictor(label=label,
                             eval_metric='root_mean_squared_error').fit(
    train_data.drop(columns=[id]),
    hyperparameters='multimodal',
    presets='best_quality',
    #auto_stack=True,
    num_stack_levels=1,
    num_bag_folds=9,
    refit_full=True,
    set_best_to_refit_full=True
    )

import pandas as pd
preds = predictor.predict(train_data)
#preds.to_csv('try.csv', index=False, header=False)
#test_data = TabularDataset('test.csv')
#preds = predictor.predict(test_data.drop(columns=[id]))
#submission = pd.DataFrame({id:test_data[id], label:preds})
#submission.to_csv('submission.csv', index=False, header=False)