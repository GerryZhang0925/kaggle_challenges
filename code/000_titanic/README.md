# url
`
[Titanic](https://www.kaggle.com/competitions/titanic)

# set up environment
```
conda create -y --force -n ag python=3.8 pip
conda activate ag
pip install "mxnet<2.0.0"
pip install autogluon
pip install kaggle
kaggle c download titanic
unzip -o titanic.zip
```
# the best model
```
Best model: "WeightedEnsemble_L2"
0.8547   = Validation score   (accuracy)
```