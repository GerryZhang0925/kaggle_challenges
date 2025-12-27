- [Model ensembling](#model-ensembling)
  - [1. Simplest ensembling technique](#1-simplest-ensembling-technique)
  - [2. Rand mean techniques](#2-rand-mean-techniques)
  - [3. Model stacking](#3-model-stacking)

# Model ensembling
Model ensembling and model blending are often used interchangeably. Usually model blending gives prediction with a simple averaging or weighted averaging scheme, and the model ensembling is a broader term that encompasses various techniques for combining the predictions of multiple base models.

Model ensembling techniques include methods such as bagging, boosting and stacking. Model bagging is used in random forest models and boosting models, it trains multiple base models independently on different subsets of the training data.

## 1 Simplest ensembling technique
Simple averaging is the simplest ensembling for regression/classification problems. and max voting is the simplest way for classification problems.
```
import numpy as np

def mean_predictions(probas):
    return np.mean(probas, axis=1)

def max_voting(preds):
    idxs = np.argmax(preds, axis=1)
    return np.take_along_axis(preds, idxs[:, None], axis=1)
```

## 2. Rand mean techniques
It is a useful method when AUC is used for evaluation.
```
def rank_mean(probas):
    # 创建空列表ranked存储每个类别概率值排名
    ranked = []
    # 遍历概率值每一列（每个类别的概率值）
    for i in range(probas.shape[1]):
        # 当前列概率值排名，rank_data是排名结果
        rank_data = stats.rankdata(probas[:, i])
        # 将当前列排名结果添加到ranked列表中
        ranked.append(rank_data)
        # 将ranked列表中排名结果按列堆叠，形成二维数组
        ranked = np.column_stack(ranked)
    # 沿着第二个维度（列）计算样本排名平均值
    return np.mean(ranked, axis=1)
```
The actual way to optimize the coefficients for multiple predictions from base models is to separate the training dataset into several folds and evaluate the coefficients on each fold.
```
import numpy as np
from functools import partial
from scipy.optimize import fmin
from sklearn import metrics

class OptimizeAUC:
    def _init _(self):
    # 初始化系数
        self.coef_ = 0

    def _auc(self, coef, X, y):
        # 对输入数据乘以系数
        x_coef = X * coef
        # 计算每个样本预测值
        predictions = np.sum(x_coef, axis=1)
        # 计算AUC分数
        auc_score = metrics.roc_auc_score(y, predictions)
        # 返回负AUC以便最小化
        return -1.0 * auc_score

    def fit(self, X, y):
        # 创建带有部分参数的目标函数
        loss_partial = partial(self._auc, X=X, y=y)
        # 初始化系数
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        # 使用fmin函数优化AUC目标函数，找到最优系数
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        # 对输入数据乘以训练好的系数
        x_coef = X * self.coef_
        # 计算每个样本预测值
        predictions = np.sum(x_coef, axis=1)
        # 返回预测结果
        return predictions
```
```
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

# 生成一个分类数据集
X, y = make_classification(n_samples=10000, n_features=25)

# 划分数据集为两个交叉验证折叠
xfold1, xfold2, yfold1, yfold2 = model_selection.train_test_split(X, y,
                                                                  test_size=0.5,
                                                                  stratify=y)
# 初始化三个不同的分类器
logreg = linear_model.LogisticRegression()
rf = ensemble.RandomForestClassifier()
xgbc = xgb.XGBClassifier()

# 使用第一个折叠数据集训练分类器
logreg.fit(xfold1, yfold1)
rf.fit(xfold1, yfold1)
xgbc.fit(xfold1, yfold1)

# 对第二个折叠数据集进行预测
pred_logreg = logreg.predict_proba(xfold2)[:, 1]
pred_rf = rf.predict_proba(xfold2)[:, 1]
pred_xgbc = xgbc.predict_proba(xfold2)[:, 1]

# 计算平均预测结果
avg_pred = (pred_logreg + pred_rf + pred_xgbc) / 3
fold2_preds = np.column_stack((pred_logreg, pred_rf, pred_xgbc, avg_pred))

# 计算每个模型的AUC分数并打印
aucs_fold2 = []
for i in range(fold2_preds.shape[1]):
    auc = metrics.roc_auc_score(yfold2, fold2_preds[:, i])
    aucs_fold2.append(auc)

print(f"Fold-2: LR AUC = {aucs_fold2[0]}")
print(f"Fold-2: RF AUC = {aucs_fold2[1]}")
print(f"Fold-2: XGB AUC = {aucs_fold2[2]}")
print(f"Fold-2: Average Pred AUC = {aucs_fold2[3]}")

# 重新初始化分类器
logreg = linear_model.LogisticRegression()
rf = ensemble.RandomForestClassifier()
xgbc = xgb.XGBClassifier()

# 使用第二个折叠数据集训练分类器
logreg.fit(xfold2, yfold2)
rf.fit(xfold2, yfold2)
xgbc.fit(xfold2, yfold2)

# 对第一个折叠数据集进行预测
pred_logreg = logreg.predict_proba(xfold1)[:, 1]
pred_rf = rf.predict_proba(xfold1)[:, 1]
pred_xgbc = xgbc.predict_proba(xfold1)[:, 1]

# 计算平均预测结果
avg_pred = (pred_logreg + pred_rf + pred_xgbc) / 3
fold1_preds = np.column_stack((pred_logreg, pred_rf, pred_xgbc, avg_pred))

# 计算每个模型的AUC分数并打印
aucs_fold1 = []
for i in range(fold1_preds.shape[1]):
auc = metrics.roc_auc_score(yfold1, fold1_preds[:, i])
aucs_fold1.append(auc)
print(f"Fold-1: LR AUC = {aucs_fold1[0]}")
print(f"Fold-1: RF AUC = {aucs_fold1[1]}")
print(f"Fold-1: XGB AUC = {aucs_fold1[2]}")
print(f"Fold-1: Average prediction AUC = {aucs_fold1[3]}")

# 初始化AUC优化器
opt = OptimizeAUC()
# 使用第一个折叠数据集的预测结果来训练优化器
opt.fit(fold1_preds[:, :-1], yfold1)
# 使用优化器对第二个折叠数据集的预测结果进行优化
opt_preds_fold2 = opt.predict(fold2_preds[:, :-1])
auc = metrics.roc_auc_score(yfold2, opt_preds_fold2)
print(f"Optimized AUC, Fold 2 = {auc}")
print(f"Coefficients = {opt.coef_}")
# 初始化AUC优化器
opt = OptimizeAUC()

# 使用第二个折叠数据集的预测结果来
opt.fit(fold2_preds[:, :-1], yfold2)

# 使用优化器对第一个折叠数据集的预测结果进行优化
opt_preds_fold1 = opt.predict(fold1_preds[:, :-1])
auc = metrics.roc_auc_score(yfold1, opt_preds_fold1)
print(f"Optimized AUC, Fold 1 = {auc}")
print(f"Coefficients = {opt.coef_}")
```

# 3. Model stacking
A method to cascade one L1 model (base models) to another L2 model (meta-model). 
- 1st step: Separate the training data set into several folds, and train a set of L1 models with them. 
- 2nd step: Ensemble those L1 models and use their outputs as features of a set of L2 models.
- 3rd step: Train the L2 models along with the true labels.
