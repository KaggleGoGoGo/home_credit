# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import gc
from xgboost.sklearn import XGBClassifier
import xgboost as xgb


def data_prepare(num=None):
    train = pd.read_csv('../data/handled/train.csv', nrows=num, index_col=0)
    test = pd.read_csv('../data/handled/test.csv', nrows=num, index_col=0)
    y_train = pd.read_csv('../data/handled/y_train.csv', nrows=num, header=-1, index_col=0)
    return train, test, y_train


def output_result(test_id, test_prob, sid=''):
    result = pd.DataFrame(np.column_stack((test_id, test_prob)))
    result.columns = ['SK_ID_CURR', 'TARGET']
    result['SK_ID_CURR'] = result['SK_ID_CURR'].astype('int')
    result.to_csv('./submission/submission_' + str(sid) + '.csv', header=True, index=False)


def show_importance(model, num=20, height=0.8):
    xgb.plot_importance(model, max_num_features=num, height=0.8)
    plt.show()


def xgb_k_folder_cv(params, xgtrain, fold=5, seed=918):
    cv = xgb.cv(params, xgtrain, metrics='auc', early_stopping_rounds=50,
                nfold=fold, seed=seed)
    return cv ## ['test-auc-mean'].values[-1]


def xgb_no_feature_select(train: pd.DataFrame, test: pd.DataFrame, y_train):
    params = {
        'silent': 1,
        'nthread': 4,
        'eval_metric': 'auc',
        'verbose_eval': True,
        'seed': 918,
        'alpha': 9.6523,
        'cosample_bytree': 0.9604,
        'eta': 0.1171,
        'gamma': 0.179,
        'max_depth': 7,
        'min_child_weight': 13,
        'subsample': 0.9609
    }
    xgtrain = xgb.DMatrix(train, label=y_train)
    cv = xgb_k_folder_cv(params, xgtrain)
    model = XGBClassifier(**params)
    model.fit(train, y_train)
    y_predict = model.predict_proba(test)
    return model, y_predict, cv


def xgb_feature_select(train, test, y_train, importance=None, top_num=None):
    pass


def xgb_unbalance_handle(train, test):
    pass


def models_stack(trian, test):
    pass


if __name__ == '__main__':
    df_train, df_test, y_train = data_prepare()
    model, y_predict, cv = xgb_no_feature_select(df_train, df_test, y_train)
    print(cv)
    output_result(df_test.index, y_predict[:, 1], '20180707_2')


