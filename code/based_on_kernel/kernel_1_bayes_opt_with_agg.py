
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from bayes_opt import  BayesianOptimization

warnings.simplefilter(action='ignore', category=FutureWarning)




def get_df(debug=False):
    df = pd.read_csv('../../data/handled/kernel_1/df_all_add_agg.csv', index_col=0, header=0)
    return df
    

def get_cls_result(
                   learning_rate,
                   num_leaves,
                   colsample_bytree,
                   subsample,
                   max_depth,
                   reg_alpha,
                   reg_lambda,
                   min_split_gain,
                   min_child_weight
                  ):
    global df, num_folds
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=47)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=8,
            n_estimators=10000,
            learning_rate=learning_rate,
            num_leaves=int(num_leaves),
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            max_depth=int(max_depth),
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            min_child_weight=int(min_child_weight),
            verbose=-1,
        )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=1000, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    return roc_auc_score(train_df['TARGET'], oof_preds)


if __name__ == "__main__":
    df = get_df()
    num_folds = 5
    BO = BayesianOptimization(get_cls_result,
            {
               'learning_rate': (0.01, 0.5),
               'num_leaves': (30, 120),
               'colsample_bytree': (0.5, 1),
               'subsample': (0.8, 1),
               'max_depth': (5, 15),
               'reg_alpha': (0, 10),
                'reg_lambda':(0, 10),
               'min_split_gain': (0, 10),
               'min_child_weight': (1, 50),
            })

    BO.explore({
               'learning_rate': [0.01, 0.02, 0.1],
               'num_leaves': [20, 32, 50],
               'colsample_bytree': [0.5, 0.95, 0.99],
               'subsample': [0.8, 0.87, 0.95],
               'max_depth': [5, 8, 15],
               'reg_alpha': [0.04, 0.1, 0.2],
                'reg_lambda':[0.073, 0.2, 0.5],
               'min_split_gain': [0.02224, 0.1, 0.2],
               'min_child_weight': [20, 40, 50],
    })

    BO.maximize(init_points=5, n_iter=30)
    print('-' * 53)
    print('Final Results')
    print('LGB: %f' % BO.res['max']['max_val'])
