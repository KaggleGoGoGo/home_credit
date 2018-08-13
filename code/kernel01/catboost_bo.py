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
from catboost import CatBoostClassifier
from bayes_opt import BayesianOptimization

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


org_dic = {'Advertising': 2,
 'Agriculture': 3,
 'Bank': 0,
 'Business Entity Type 1': 2,
 'Business Entity Type 2': 2,
 'Business Entity Type 3': 2,
 'Cleaning': 3,
 'Construction': 3,
 'Culture': 0,
 'Electricity': 1,
 'Emergency': 1,
 'Government': 1,
 'Hotel': 1,
 'Housing': 1,
 'Industry: type 1': 3,
 'Industry: type 10': 1,
 'Industry: type 11': 2,
 'Industry: type 12': 0,
 'Industry: type 13': 3,
 'Industry: type 2': 1,
 'Industry: type 3': 3,
 'Industry: type 4': 3,
 'Industry: type 5': 1,
 'Industry: type 6': 1,
 'Industry: type 7': 2,
 'Industry: type 8': 3,
 'Industry: type 9': 1,
 'Insurance': 0,
 'Kindergarten': 1,
 'Legal Services': 1,
 'Medicine': 1,
 'Military': 0,
 'Mobile': 2,
 'Other': 1,
 'Police': 0,
 'Postal': 2,
 'Realtor': 3,
 'Religion': 0,
 'Restaurant': 3,
 'School': 0,
 'Security': 2,
 'Security Ministries': 0,
 'Self-employed': 3,
 'Services': 1,
 'Telecom': 1,
 'Trade: type 1': 2,
 'Trade: type 2': 1,
 'Trade: type 3': 3,
 'Trade: type 4': 0,
 'Trade: type 5': 1,
 'Trade: type 6': 0,
 'Trade: type 7': 2,
 'Transport: type 1': 0,
 'Transport: type 2': 1,
 'Transport: type 3': 3,
 'Transport: type 4': 2,
 'University': 0,
 'XNA': 0}



# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../../data/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../../data/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    # 我添加的
    df['DAYS_EMPLOYED_365243'] = df['DAYS_EMPLOYED'].map(lambda x: 1 if x == 365243 else 0 if not pd.isnull(x) else np.nan)
    #for col in Nan_matters_col:
    #    df[col+'_NAN'] = df[col].map(lambda x: 1 if  pd.isnull(x) else 0)
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']).replace(np.inf, np.nan)
    df['NEW_CREDIT_TO_GOODS_RATIO'] = (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']).replace(np.inf, np.nan)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = (df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])).replace(np.inf, np.nan)
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = (df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = (df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])).replace(np.inf, np.nan)
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = (df['OWN_CAR_AGE'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['NEW_CAR_TO_EMPLOY_RATIO'] = (df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']).replace(np.inf, np.nan)
    df['NEW_PHONE_TO_BIRTH_RATIO'] = (df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = (df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']).replace(np.inf, np.nan)
    df['NEW_CREDIT_TO_INCOME_RATIO'] = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']).replace(np.inf, np.nan)
    
#     # 我的变量 start
    df['have_car_and_house'] =  list(map(lambda x, y: 1 if x=='Y' and y=='Y' else 0, df['FLAG_OWN_CAR'], df['FLAG_OWN_REALTY']))
    df['income_credict_ratio'] = (df['AMT_INCOME_TOTAL'] /df['AMT_CREDIT']).replace(np.inf, np.nan)
    edu_map = {'Academic degree':0, 'Higher education':1, 'Incomplete higher':2, 'Secondary / secondary special':3, 'Lower secondary':4}
    df['education_type'] = df['NAME_EDUCATION_TYPE'].map(edu_map)
    df['credit_age_ratio'] = (df['AMT_CREDIT'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['days_start_work'] = (df['DAYS_BIRTH'] - df['DAYS_EMPLOYED'])
    df['days_start_buy_car'] = (df['DAYS_BIRTH'] - df['OWN_CAR_AGE'])
    df['mobile'] = list(map(int, np.logical_or(df['FLAG_EMP_PHONE'] , np.logical_or(df['FLAG_WORK_PHONE'], df['FLAG_PHONE']))))
    df['adult_num'] = list(map(lambda x, y: 1 if x-y==1 and x != 1 else 0, df['CNT_FAM_MEMBERS'], df['CNT_CHILDREN']))
    df['region_rationg'] = df['REGION_RATING_CLIENT'] * df['REGION_RATING_CLIENT_W_CITY']
    credit_by_org = df[['AMT_CREDIT', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_CREDIT']
    df['credict_by_org'] =  df['ORGANIZATION_TYPE'].map(credit_by_org)
    df['org_is_nan'] = df['ORGANIZATION_TYPE'].map(lambda x: 1 if x == 'XNA' else 0)
    df['org_id'] = df['ORGANIZATION_TYPE'].map(org_dic)
    df['ext_source'] = np.nanmedian(df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    df['ext_source2'] = np.nansum(df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    df['ext_source3'] = np.nanmin(df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)


    df['app missing'] = df.isnull().sum(axis = 1).values
    df['app EXT_SOURCE_1 * EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['app EXT_SOURCE_1 * EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_2 * EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_1 * DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_2 * DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_3 * DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_1 / DAYS_BIRTH'] = (df['EXT_SOURCE_1'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['app EXT_SOURCE_2 / DAYS_BIRTH'] = (df['EXT_SOURCE_2'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['app EXT_SOURCE_3 / DAYS_BIRTH'] = (df['EXT_SOURCE_3'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)

    df['app AMT_CREDIT - AMT_GOODS_PRICE'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['app AMT_CREDIT / AMT_GOODS_PRICE'] = (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']).replace(np.inf, np.nan)
    df['app AMT_CREDIT / AMT_ANNUITY'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']).replace(np.inf, np.nan)
    df['app AMT_CREDIT / AMT_INCOME_TOTAL'] = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']).replace(np.inf, np.nan)

    df['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / 12. - df['AMT_ANNUITY']
    df['app AMT_INCOME_TOTAL / AMT_ANNUITY'] = (df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']).replace(np.inf, np.nan)
    df['app AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = df['AMT_INCOME_TOTAL'] - df['AMT_GOODS_PRICE']
    df['app AMT_INCOME_TOTAL / CNT_FAM_MEMBERS'] = (df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']).replace(np.inf, np.nan)
    df['app AMT_INCOME_TOTAL / CNT_CHILDREN'] = (df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])).replace(np.inf, np.nan)

    df['app most popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
                        .isin([225000, 450000, 675000, 900000]).map({True: 1, False: 0})
    df['app popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
                        .isin([1125000, 1350000, 1575000, 1800000, 2250000]).map({True: 1, False: 0})

    df['app OWN_CAR_AGE / DAYS_BIRTH'] = (df['OWN_CAR_AGE'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['app OWN_CAR_AGE / DAYS_EMPLOYED'] = (df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']).replace(np.inf, np.nan)

    df['app DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH'] = (df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['app DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED'] = (df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']).replace(np.inf, np.nan)
    df['app DAYS_EMPLOYED - DAYS_BIRTH'] = (df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']).replace(np.inf, np.nan)
    df['app DAYS_EMPLOYED / DAYS_BIRTH'] = (df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']).replace(np.inf, np.nan)

    df['app CNT_CHILDREN / CNT_FAM_MEMBERS'] = (df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']).replace(np.inf, np.nan)
    
    #  我的变量 end
    
    
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../../data/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../../data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    

    def weighted_mean(x, balance):
        weight = np.sqrt(-1 / balance[x.index])
        return np.nanmean(np.multiply(x, weight))
    
    bb['bb_not_zero'] = bb.loc[:, ['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5']].sum(axis=1)
    # bb['bb_c_ratio'] =bb['STATUS_C'] / bb['BUREAU_BAL_COUNT']
    
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size'],
                      'bb_not_zero':['max', 'mean']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
        
    bb = pd.merge(bb, bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU', how='left')

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    temp = [i for i in bb_agg.columns if 'STATUS_' in i]
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    bureau.loc[bureau['AMT_ANNUITY'] > .8e8, 'AMT_ANNUITY'] = np.nan
    bureau.loc[bureau['AMT_CREDIT_SUM'] > 3e8, 'AMT_CREDIT_SUM'] = np.nan
    bureau.loc[bureau['AMT_CREDIT_SUM_DEBT'] > 1e8, 'AMT_CREDIT_SUM_DEBT'] = np.nan
    bureau.loc[bureau['AMT_CREDIT_MAX_OVERDUE'] > .8e8, 'AMT_CREDIT_MAX_OVERDUE'] = np.nan
    bureau.loc[bureau['DAYS_ENDDATE_FACT'] < -10000, 'DAYS_ENDDATE_FACT'] = np.nan
    bureau.loc[(bureau['DAYS_CREDIT_UPDATE'] > 0) | (bureau['DAYS_CREDIT_UPDATE'] < -40000), 'DAYS_CREDIT_UPDATE'] = np.nan
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < -10000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    
    bureau['app_credit_annuity_ratio'] = (bureau['AMT_CREDIT_SUM'] /  bureau['AMT_ANNUITY']).replace(np.inf, np.nan)
    bureau['app_credit_debt_ratio'] = (bureau['AMT_CREDIT_SUM'] /  bureau['AMT_CREDIT_SUM_DEBT']).replace(np.inf, np.nan)
    bureau['app_credit_limit_ratio'] = (bureau['AMT_CREDIT_SUM'] /  bureau['AMT_CREDIT_SUM_LIMIT']).replace(np.inf, np.nan)
    bureau['app_credit_overdue_ratio'] = (bureau['AMT_CREDIT_SUM'] /  bureau['AMT_CREDIT_SUM_OVERDUE']).replace(np.inf, np.nan)
    bureau['weighted_credit'] = (bureau['AMT_CREDIT_SUM'] / np.sqrt(-bureau['DAYS_CREDIT'])).replace(np.inf, np.nan)
    

    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_OVERDUE']
    
    
    bureau['bureau DAYS_CREDIT - CREDIT_DAY_OVERDUE'] = bureau['DAYS_CREDIT'] - bureau['CREDIT_DAY_OVERDUE']
    bureau['bureau DAYS_CREDIT - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['bureau DAYS_CREDIT - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau DAYS_CREDIT_ENDDATE - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT_ENDDATE']
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var', 'min'],
        'DAYS_CREDIT_ENDDATE': [ 'mean', 'max'],
        'DAYS_CREDIT_UPDATE': ['mean', 'min'],
        'DAYS_ENDDATE_FACT':['mean', 'max', 'min', 'var'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max', 'var'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean','median'],
        'CNT_CREDIT_PROLONG': ['sum','mean','max'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'app_credit_annuity_ratio':['mean', 'median'],
        'app_credit_debt_ratio':['mean', 'median'],
        'app_credit_limit_ratio':['mean', 'median'],
        'app_credit_overdue_ratio':['mean', 'median'],
        'weighted_credit':['mean', 'sum', 'median'],
        'bb_not_zero_MEAN':['mean'],
        'bb_not_zero_MAX':['mean'],
        
    }
    
    
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for col in bureau.columns:
        if 'bureau' in col:
            cat_aggregations[col] = ['mean', 'max', 'min', 'std', 'median']
    for cat in bureau_cat: cat_aggregations[cat] = ['mean', 'sum']
    for cat in temp:
        cat_aggregations[cat] = ['mean', 'std', 'median']
    #for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean', 'sum', mode]
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../../data/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    
    #prev['DAYS_LAST_DUE_ISNAN'] = prev['DAYS_LAST_DUE'].map(lambda x: 1 if x == 365243 else 0 if not pd.isnull(x) else np.nan)
    #prev['DAYS_FIRST_DRAWING_ISNAN'] = prev['DAYS_FIRST_DRAWING'].map(lambda x: 1 if x == 365243 else 0 if not pd.isnull(x) else np.nan)
    #prev['DAYS_FIRST_DUE_ISNAN'] = prev['DAYS_FIRST_DUE'].map(lambda x: 1 if x == 365243 else 0 if not pd.isnull(x) else np.nan)
    #prev['DAYS_LAST_DUE_1ST_VERSION_ISNAN'] = prev['DAYS_LAST_DUE_1ST_VERSION'].map(lambda x: 1 if x == 365243 else 0 if not pd.isnull(x) else np.nan)
    #prev['DAYS_TERMINATION_ISNAN'] = prev['DAYS_TERMINATION'].map(lambda x: 1 if x == 365243 else 0 if not pd.isnull(x) else np.nan)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    # my features start
    prev['prev missing'] = prev.isnull().sum(axis = 1).values
    prev['prev AMT_APPLICATION / AMT_CREDIT'] = (prev['AMT_APPLICATION'] / prev['AMT_CREDIT']).replace(np.inf, np.nan)
    prev['prev AMT_APPLICATION - AMT_CREDIT'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['prev AMT_APPLICATION - AMT_GOODS_PRICE'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['prev AMT_GOODS_PRICE - AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] - prev['AMT_CREDIT']
    prev['prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE'] = prev['DAYS_FIRST_DRAWING'] - prev['DAYS_FIRST_DUE']
    prev['prev DAYS_TERMINATION less -500'] = (prev['DAYS_TERMINATION'] < -500).astype(int)
    
    prev['prev AMT_CREDIT / AMT_ANNUITY'] = (prev['AMT_CREDIT'] / prev['AMT_ANNUITY']).replace(np.inf, np.nan)
    prev['prev AMT_APPLICATION / AMT_ANNUITY'] = (prev['AMT_APPLICATION'] / prev['AMT_ANNUITY']).replace(np.inf, np.nan)
    prev['prev AMT_APPLICATION - AMT_ANNUITY'] = prev['AMT_APPLICATION'] - prev['AMT_ANNUITY']
    prev['prev AMT_APPLICATION / AMT_DOWN_PAYMENT'] = (prev['AMT_APPLICATION'] / prev['AMT_DOWN_PAYMENT']).replace(np.inf, np.nan)
    prev['prev AMT_APPLICATION - AMT_DOWN_PAYMENT'] = prev['AMT_APPLICATION'] - prev['AMT_DOWN_PAYMENT']
    prev['prev RATE_INTEREST_PRIMARY - RATE_INTEREST_PRIVILEGED'] = prev['RATE_INTEREST_PRIMARY'] - prev['RATE_INTEREST_PRIVILEGED']
    prev['prev AMT_APPLICATION / CNT_PAYMENT'] = (prev['AMT_APPLICATION'] / prev['CNT_PAYMENT']).replace(np.inf, np.nan)
    
    
    # my features end
    
    
    # Add feature: value ask / value received percentage
    #prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean', 'min'],
        'AMT_APPLICATION': [ 'max','mean', 'min', 'size'],
        'AMT_CREDIT': [ 'max', 'mean', 'min', 'std'],
        #'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean', 'min'],
        'AMT_GOODS_PRICE': [ 'max', 'mean', 'min', 'sum'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean', 'min'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean', 'min'],
        'DAYS_DECISION': [ 'max', 'mean', 'min'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'RATE_INTEREST_PRIMARY': [ 'max', 'mean', 'min'],
        'RATE_INTEREST_PRIVILEGED': [ 'max', 'mean', 'min'],
    }
#         'DAYS_LAST_DUE_ISNAN':['mean', 'sum'], 
#         'DAYS_FIRST_DRAWING_ISNAN':['mean', 'sum'], 
#         'DAYS_FIRST_DUE_ISNAN':['mean', 'sum'], 
#         'DAYS_LAST_DUE_1ST_VERSION_ISNAN':['mean', 'sum'], 
#         'DAYS_TERMINATION_ISNAN':['mean', 'sum'], 
    for i in prev.columns:
        if 'prev' in i:
            num_aggregations[i] = ['mean'] #  'min', 'max', 'size', 'mean', 'var', 'sum']
    
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg



# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../../data/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    
    # my feature start
    pos['pos installment_sum'] = pos['CNT_INSTALMENT'] + pos['CNT_INSTALMENT_FUTURE']
    pos['pos dpd_sum'] = pos['SK_DPD'] + pos['SK_DPD_DEF']
    
    # my feature end
    
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size', 'min'],
        'SK_DPD': [ 'max', 'mean'],
        'SK_DPD_DEF': [ 'max', 'mean'],
        'pos installment_sum': ['sum', 'max', 'mean'],
        'CNT_INSTALMENT': ['sum', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['sum', 'max', 'mean'],
        'pos dpd_sum': ['sum', 'max', 'mean'],
    }
    for cat in cat_cols:
        aggregations[cat] =[ 'max', 'mean', 'sum']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    pos_agg_prev = pos.groupby('SK_ID_PREV').agg(aggregations)
    pos_agg_prev.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev.columns.tolist()])
    pos_agg_prev = pd.merge(pos_agg_prev, pos[['SK_ID_PREV', 'SK_ID_CURR']], left_index=True, right_on='SK_ID_PREV',how='left')
    pos_agg_prev = pos_agg_prev.drop(columns=['SK_ID_PREV'])
    aggregation2 = {}
    for i in pos_agg_prev.columns:
        aggregation2[i] = ['mean', 'max']
    pos_agg_prev2 = pos_agg_prev.groupby('SK_ID_CURR').agg(aggregation2)
    pos_agg_prev2.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev2.columns.tolist()])
    pos_agg_prev2 = pos_agg_prev2.drop(columns=['POS_SK_ID_CURR_MEAN', 'POS_SK_ID_CURR_MAX'])
    
    pos_agg = pos_agg.join(pos_agg_prev2, how='left')
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    del pos, pos_agg_prev, pos_agg_prev2
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../../data/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = (ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']).replace(np.inf, np.nan)
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # my feature start
    ins['ins AMT_INSTALMENT==AMT_PAYMENT'] = (ins['AMT_INSTALMENT'] == ins['AMT_PAYMENT']).astype('int')
    ins['ins DPD_ratio'] = (ins['DAYS_ENTRY_PAYMENT'] / ins['DAYS_INSTALMENT']).replace(np.inf, np.nan)
    ins['ins NUM_INSTALMENT_VERSION=1'] = ins['NUM_INSTALMENT_VERSION'].map(lambda x: 1 if x == 1 else 0 if not pd.isnull(x) else np.nan)
    ins['ins NUM_INSTALMENT_VERSION=0'] = ins['NUM_INSTALMENT_VERSION'].map(lambda x: 1 if x == 0 else 0 if not pd.isnull(x) else np.nan)
    #ins['ins AMT_PAYMENT / DPD'] =(ins['AMT_PAYMENT'] / ins['DPD']).replace(np.inf, np.nan)
    
    # my feature end
    
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min' ],
        'DBD': ['max', 'mean', 'sum','min'],
        'PAYMENT_PERC': [ 'max','mean','min'],
        'PAYMENT_DIFF': [ 'max','mean','min'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
        'NUM_INSTALMENT_NUMBER': ['max'],
        'NUM_INSTALMENT_VERSION': ['max'],
    }
    for col in ins.columns:
        if 'ins' in col:
            aggregations[col] = ['mean', 'max', 'min']
    
    for cat in cat_cols:
        aggregations[cat] =[ 'max', 'mean', 'sum']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    
    ins_agg_prev = ins.groupby('SK_ID_PREV').agg(aggregations)
    ins_agg_prev.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg_prev.columns.tolist()])
    ins_agg_prev = pd.merge(ins_agg_prev, ins[['SK_ID_PREV', 'SK_ID_CURR']], left_index=True, right_on='SK_ID_PREV',how='left')
    ins_agg_prev = ins_agg_prev.drop(columns=['SK_ID_PREV'])
    aggregation2 = {}
    for i in ins_agg_prev.columns:
        aggregation2[i] = ['mean', 'max', 'min']
    ins_agg_prev2 = ins_agg_prev.groupby('SK_ID_CURR').agg(aggregation2)
    ins_agg_prev2.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg_prev2.columns.tolist()])
    ins_agg_prev2 = ins_agg_prev2.drop(columns=['INS_SK_ID_CURR_MEAN', 'INS_SK_ID_CURR_MAX','INS_SK_ID_CURR_MIN'])
    
    ins_agg = ins_agg.join(ins_agg_prev2, how='left')
    
    
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg




# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../../data/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    
    
    # my feature start
    cc['cc AMT_BALANCE / MONTHS_BALANCE'] = (cc['AMT_BALANCE'] / cc['MONTHS_BALANCE']).replace(np.inf, np.nan)
    cc['cc AMT_CREDIT_LIMIT_ACTUAL / AMT_BALANCE'] = (cc['AMT_CREDIT_LIMIT_ACTUAL'] / cc['AMT_BALANCE']).replace(np.inf, np.nan)
    cc['cc AMOUNT_DRAWING_SUM'] = cc['AMT_DRAWINGS_ATM_CURRENT'] + cc['AMT_DRAWINGS_CURRENT'] + \
                            cc['AMT_DRAWINGS_OTHER_CURRENT'] + cc['AMT_DRAWINGS_POS_CURRENT']
    cc['cc AMT_BALANCE / AMT_INST_MIN_REGULARITY'] = (cc['AMT_BALANCE'] / cc['AMT_INST_MIN_REGULARITY']).replace(np.inf, np.nan)
    cc['cc AMT_BALANCE / AMT_PAYMENT_CURRENT'] = (cc['AMT_BALANCE'] / cc['AMT_PAYMENT_CURRENT']).replace(np.inf, np.nan)
    cc['cc AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY'] = (cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']).replace(np.inf, np.nan)
    cc['cc AMT_PAYMENT_CURRENT / AMT_PAYMENT_TOTAL_CURRENT'] = (cc['AMT_PAYMENT_CURRENT'] / cc['AMT_PAYMENT_TOTAL_CURRENT']).replace(np.inf, np.nan)
    cc['cc AMT_RECIVABLE / AMT_RECEIVABLE_PRINCIPAL'] = (cc['AMT_RECIVABLE'] / cc['AMT_RECEIVABLE_PRINCIPAL']).replace(np.inf, np.nan)
    cc['cc AMT_RECIVABLE / AMT_TOTAL_RECEIVABLE'] = (cc['AMT_RECIVABLE'] / cc['AMT_TOTAL_RECEIVABLE']).replace(np.inf, np.nan)
    
    
    cc['cc AMT_DRAWINGS_ATM_CURRENT / CNT_DRAWINGS_ATM_CURRENT'] = (cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['CNT_DRAWINGS_ATM_CURRENT']).replace(np.inf, np.nan)
    cc['cc AMT_DRAWINGS_CURRENT / CNT_DRAWINGS_CURRENT'] = (cc['AMT_DRAWINGS_CURRENT'] / cc['CNT_DRAWINGS_CURRENT']).replace(np.inf, np.nan)
    cc['cc AMT_DRAWINGS_OTHER_CURRENT / CNT_DRAWINGS_OTHER_CURRENT'] = (cc['AMT_DRAWINGS_OTHER_CURRENT'] / cc['CNT_DRAWINGS_OTHER_CURRENT']).replace(np.inf, np.nan)
    cc['cc AMT_DRAWINGS_POS_CURRENT / CNT_DRAWINGS_POS_CURRENT'] = (cc['AMT_DRAWINGS_POS_CURRENT'] / cc['CNT_DRAWINGS_POS_CURRENT']).replace(np.inf, np.nan)
    
    cc['cc CNT_DRAWING_SUM'] = cc['CNT_DRAWINGS_ATM_CURRENT'] + cc['CNT_DRAWINGS_CURRENT'] + \
                            cc['CNT_DRAWINGS_OTHER_CURRENT'] + cc['CNT_DRAWINGS_POS_CURRENT']
    
    cc['cc AMT_DRAWING / CNT_DRAWING'] = (cc['cc AMOUNT_DRAWING_SUM']  / cc['cc CNT_DRAWING_SUM']).replace(np.inf, np.nan)
    # my feature end
   
    cc_agg_prev = cc.groupby('SK_ID_PREV').agg([ 'max', 'mean'])
    cc_agg_prev.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg_prev.columns.tolist()])
    cc_agg_prev = pd.merge(cc_agg_prev, cc[['SK_ID_PREV', 'SK_ID_CURR']], left_index=True, right_on='SK_ID_PREV',how='left')
    cc_agg_prev = cc_agg_prev.drop(columns=['SK_ID_PREV'])
    cc_agg_prev2 = cc_agg_prev.groupby('SK_ID_CURR').agg(['mean'])
    cc_agg_prev2.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg_prev2.columns.tolist()])
    #cc_agg_prev2 = cc_agg_prev2.drop(columns=['CC_SK_ID_CURR_MEAN', 'CC_SK_ID_CURR_MAX','CC_SK_ID_CURR_MIN'])
    
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'min', 'sum'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    
    cc_agg = cc_agg.join(cc_agg_prev2, how='left')
    
    
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg





def trans(train, test1):
    feat = []
    for col in train.columns:
        s1 = train[col].isnull().sum()
        s2 = test1[col].isnull().sum()
        
        if (s1 == 0 and s2 != 0):
            pass
        else:
            feat.append(col)
    return feat


def kfold_catboost(learning_rate, depth, l2_leaf_reg, model_size_reg, rsm, subsample):
    global train_df
    num_folds = 5
    folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    # sub_preds = np.zeros(test_df.shape[0])
    # feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        feat = trans(train_x, valid_x)
        train_x = train_x[feat]
        valid_x = valid_x[feat]        
        
        cat_params = {
            'iterations': 5000,
            'learning_rate': learning_rate,
            'depth': int(depth),
            'l2_leaf_reg': l2_leaf_reg,
            'bootstrap_type': 'Bernoulli',
            'subsample': subsample,
            'model_size_reg': model_size_reg,
            'rsm': rsm,
            'scale_pos_weight': 5,
            'eval_metric': 'AUC',
            'od_type': 'Iter',
            'allow_writing_files': False,
            'thread_count':8
        }
        clf = CatBoostClassifier(**cat_params)

        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose= False, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        # sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    return roc_auc_score(train_df['TARGET'], oof_preds)


debug = False
num_rows = 10000 if debug else None
df = application_train_test(num_rows)
with timer("Process bureau and bureau_balance"):
    bureau = bureau_and_balance(num_rows)
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
with timer("Process previous_applications"):
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()
with timer("Process POS-CASH balance"):
    pos = pos_cash(num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()
with timer("Process installments payments"):
    ins = installments_payments(num_rows)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()
with timer("Process credit card balance"):
    cc = credit_card_balance(num_rows)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()
    
train_df = df[df['TARGET'].notnull()]
del df
gc.collect()

bo = BayesianOptimization(kfold_catboost, {'learning_rate': (0.005, 0.02), 
                                            'depth': (4, 10), 
                                            'l2_leaf_reg': (0.01, 0.5), 
                                            'model_size_reg': (0.01, 0.5), 
                                            'subsample': (0.8, 0.99), 
                                            'rsm': (0.8, 0.99)
                                           })
                          
gp_param={'alpha':1e-5}
bo.maximize(**gp_param)