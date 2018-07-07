# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
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


def data_prepare(num=None):
    train = pd.read_csv('../data/handled/train.csv', nrows=num)
    test = pd.read_csv('../data/handled/test.csv', nrows=num)
    return train, test


if __name__ == '__main__':
    train, test = data_prepare(1000)
    print(train.shape)

