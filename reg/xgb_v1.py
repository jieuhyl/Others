#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:07:20 2019

@author: qianqianwang
"""


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings('ignore') 
from scipy import stats
from scipy.stats import norm, skew 
from datetime import datetime

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_score, GridSearchCV


import time

df = pd.read_csv('Monday_t0000.csv')

# check missing
df.isnull().values.sum()

# numerical variables
numeric_feats = df.iloc[:,3:57].dtypes[df.iloc[:,3:57].dtypes != "object"].index
# Check the skew of all numerical features
skewed_feats = df[3:57][numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[np.abs(skewness['Skew']) >= 0.75]
print("There are {} skewed numerical features to log1p transform".format(skewness.shape[0]))


skewed_features = skewness.index
for feat in skewed_features:
    #all_data[feat] += 1
    #df[feat] = boxcox1p(df[feat], lam)
    df[feat] = np.log1p(df[feat])
    

# get mtx
y = np.log1p(df['OBO']).values
X = df.iloc[:,4:].values

# transformation
#rb = RobustScaler()
#X = rb.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# xgboost =====================================================================
mod_xgb = xgb.XGBRegressor(random_state= 1337)
mod_xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = mod_xgb.predict(X_train)
mape = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred)) / np.expm1(y_train))) * 100
#Print model report:
print ("\nXGB Model Report")
print ("MAPE : %.2f" % mape)

'''
def run_xgb(X_train, X_test, y_train, y_test):

    start_time = time.time()
    params = {
        "objective": 'reg:linear',
        "booster" : "gbtree",
        "eval_metric": "rmse",
        "eta": 0.1,
        "gamma": 0.001,
        "max_depth": 7,
        "min_child_weight": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha":0.1,
        "reg_lambda":0.1,
        "silent": 1,
        "seed": 1337
    }
    
    #num_boost_round = 500
    #early_stopping_rounds = 50

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    mod_xgb = xgb.train(params, 
                        dtrain, 
                        evals=watchlist,
                        eval_metric=["error", "logloss"],
                        num_boost_round=500, 
                        early_stopping_rounds=50, 
                        verbose_eval=100)

    print("Validating...")
    y_pred = mod_xgb.predict(xgb.DMatrix(X_test), ntree_limit=mod_xgb.best_iteration)
    score = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred)) / np.expm1(y_test))) * 100


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return score #y_pred

run_xgb(X_train, X_test, y_train, y_test)
'''

start_time = time.time()
params = {
    "objective": 'reg:linear',
    "booster" : "gbtree",
    "eval_metric": "rmse",
    "eta": 0.1,
    "gamma": 0,
    "max_depth": 3,
    "min_child_weight": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha":0.01,
    "reg_lambda":0.1,
    "silent": 1,
    "seed": 1337
}


dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_test, y_test)


watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
evals_result = {}

mod_xgb = xgb.train(params, 
                    dtrain, 
                    evals=watchlist,
                    num_boost_round=500, 
                    early_stopping_rounds=50, 
                    verbose_eval=100,
                    evals_result=evals_result)

y_pred1 = mod_xgb.predict(xgb.DMatrix(X_train), ntree_limit=mod_xgb.best_iteration)
y_pred2 = mod_xgb.predict(xgb.DMatrix(X_test), ntree_limit=mod_xgb.best_iteration)
score1 = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred1)) / np.expm1(y_train))) * 100
score2 = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred2)) / np.expm1(y_test))) * 100
print ("\nXGB Model Report")
print("train {:.2f} | valid {:.2f}".format(float(score1), float(score2)))


# hyperparameters tuning
# xgboost
cv_xgb = xgb.XGBRegressor(learning_rate=0.1, n_estimators=130, objective='reg:linear', silent=1, verbosity = 2)

parameters = { 
              'learning_rate':[0.05,0.1,0.2],
              'gamma':[0],
              'max_depth':[3,4,5], 
              'min_child_weight':[7,8,9],
              'subsample':[0.7,0.8,0.9],
              'colsample_bytree':[0.7,0.8,0.9],
              'reg_alpha':[0,0.01,0.1],
              'reg_lambda':[0,0.01,0.1]
              
              }

grid_search = GridSearchCV(estimator = cv_xgb,
                           param_grid = parameters,
                           scoring='neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# plot learning curve
epochs = len(evals_result['train']['rmse'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, evals_result['train']['rmse'], label='Train')
ax.plot(x_axis, evals_result['valid']['rmse'], label='Valid')
ax.legend()
plt.ylabel('rmse')
plt.title('XGBoost Loss')
plt.show()

# feature importance
print('Plot feature importances...')
ax = xgb.plot_importance(mod_xgb, max_num_features=10)
plt.show()


mapper = {'f{0}'.format(i): v for i, v in enumerate(df.iloc[:,4:].columns)}
mapped1 = {mapper[k]: v for k, v in mod_xgb.get_score(importance_type='weight').items()}
mapped2 = {mapper[k]: v for k, v in mod_xgb.get_score(importance_type='gain').items()}
#featureimp = pd.DataFrame(list(mapped.items()), columns=['Features', 'gain']).sort_values('gain', ascending=False)
featureimp = pd.DataFrame({'weight':pd.Series(mapped1),'gain':pd.Series(mapped2)}).rename_axis('Feature').reset_index()
print(featureimp[:10])





