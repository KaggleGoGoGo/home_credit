# 待办记录


## 现有数据集不变动 ../data/handled/train.csv



### lightgbm 手动调参


### lightgbm 全特征bayes调参


### lightgbm 最优参数基础上特征选择


```python
params_lgb = {
    'nthread': 4,
    #is_unbalance=True,
    'n_estimators' : 10000,
    'learning_rate' : 0.1171,
    #'num_leaves' : 32,
    'colsample_bytree' : 0.9604,
    'subsample' : 0.9609,
    'max_depth' : 7,
    'reg_alpha' : 9.6523,
    'reg_lambda' : 1,
    'min_split_gain' : 0.179,
    'min_child_weight' : 13,
    'metric': 'auc',
    'silent': -1,
    'verbose': -1,
    #scale_pos_weight=11
}
```

### 其他模型尝试


### 其他模型融合




## 现有数据集变动


### 人工筛选特征

### 直接尝试别人的notebook