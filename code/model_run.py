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
    xgb.plot_importance(model, max_num_features=num, height=height)
    plt.show()


def xgb_k_folder_cv(params, xgtrain, fold=5, seed=918):
    cv = xgb.cv(params, xgtrain, metrics='auc', early_stopping_rounds=50,
                nfold=fold, seed=seed)
    return cv   # ['test-auc-mean'].values[-1]


def xgb_evaluate(params,
                 xgtrain,
                 #   以下需要再次调用匿名函数封装
                 eta,
                 min_child_weight,
                 cosample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):
    params['eta'] = max(eta, 0)
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(cosample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['min_child_weight'] = int(min_child_weight)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)

    cv = xgb.cv(params, xgtrain, metrics='auc', early_stopping_rounds=50,
                nfold=5, seed=918)
    return cv['test-auc-mean'].values[-1]


def xgb_no_feature_select(train: pd.DataFrame, test: pd.DataFrame, y_train, cv=False):
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
    if cv:
        cv_res = xgb_k_folder_cv(params, xgtrain)
        print(cv_res)
    model = XGBClassifier(**params)
    model.fit(train, y_train)
    y_predict = model.predict_proba(test)
    return model, y_predict


def xgb_feature_select(train, test, y_train, importance, top_num=None, cv=False):
    if top_num:
        threshold = np.sort(importance)[-top_num-1]
    else:
        threshold = 0
    select_id = [True if i > threshold else False for i in importance]
    train = train.loc[:, select_id]
    test = test.loc[:, select_id]
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
    xgtest = xgb.DMatrix(test)
    if cv:
        cv_res = xgb_k_folder_cv(params, xgtrain)
        print(cv_res)
    model = XGBClassifier(**params)
    model.fit(train, y_train)
    y_predict = model.predict_proba(test)
    return model, y_predict


def xgb_bayes_opt(train, y_train):
    # num_rounds = 3000
    random_state = 918
    num_iter = 25
    init_points = 5
    params = {
        'silent': 1,
        'nthread': 4,
        'eval_metric': 'auc',
        'verbose_eval': True,
        'seed': random_state,
    }
    xgtrian = xgb.DMatrix(train, label=y_train)
    _xgb_evaluate = lambda a, b, c, d, e, f, g: xgb_evaluate(params, xgtrian, a, b, c, d, e, f, g)
    xgbBO = BayesianOptimization(_xgb_evaluate, {
        'eta': (0.1, 0.5),
        'min_child_weight': (1, 20),
        'cosample_bytree': (0.1, 1),
        'max_depth': (5, 15),
        'subsample': (0.5, 1),
        'gamma': (0, 10),
        'alpha': (0, 10)
    })
    xgbBO.maximize(init_points=init_points, n_iter=num_iter)
    return xgbBO


def xgb_unbalance_handle(train, test):
    pass


def models_stack(trian, test):
    pass


if __name__ == '__main__':
    df_train, df_test, y_train = data_prepare()
    model, y_predict = xgb_no_feature_select(df_train, df_test, y_train)
    model, y_predict = xgb_feature_select(df_train, df_test, y_train, model.feature_importances_, cv=True)
    output_result(df_test.index, y_predict[:, 1], '20180707_3')


