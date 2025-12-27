from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('train.csv')
id, label = 'PassengerId', 'Survived'
predictor = TabularPredictor(label=label).fit(
    train_data.drop(columns=[id]),
    hyperparameters='multimodal',
    #num_stack_levels=1, num_bag_folds=5
    )

import pandas as pd

test_data = TabularDataset('test.csv')
preds = predictor.predict(test_data.drop(columns=[id]))
submission = pd.DataFrame({id:test_data[id], label:preds})
submission.to_csv('submission.csv', index=False)
