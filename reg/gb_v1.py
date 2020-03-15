#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 00:32:56 2019

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
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
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


# gb ==========================================================================
mod_gb = GradientBoostingRegressor(random_state= 1337)
mod_gb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = mod_gb.predict(X_test)
mape_gb = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred)) / np.expm1(y_test))) * 100
#Print model report:
print ("\GB nModel Report")
print ("MAPE : %.2f" % mape_gb)


# hyperparameters tuning
def my_scorer(y_true, y_pred):
    mape = np.mean(np.abs((np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true))) * 100
    return mape
my_func = make_scorer(my_scorer, greater_is_better=False)

parameters = {'learning_rate':[0.01, 0.1, 0.2], 
              'max_depth':[3,5,7],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_samples_leaf':[1,3,5,7],
              'min_samples_split':[2,4,6,8],
              'n_estimators':[80,100,120],
              'subsample':[0.4,0.6,0.8]}
                                       
grid_search = GridSearchCV(estimator = mod_gb,
                           param_grid = parameters,
                           scoring=my_func,
                           cv = 5,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
mod_gb = GradientBoostingRegressor(learning_rate=0.1, 
                                        max_depth = 3, 
                                        max_features = 'auto',
                                        min_samples_leaf = 5, 
                                        min_samples_split = 2,
                                        n_estimators=100, 
                                        subsample=0.4,
                                        random_state= 1337 )
mod_gb.fit(X_train, y_train)

y_pred1 = mod_gb.predict(X_train)
y_pred2 = mod_gb.predict(X_test)
score1 = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred1)) / np.expm1(y_train))) * 100
score2 = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred2)) / np.expm1(y_test))) * 100
print ("\nGB Model Report")
print("train {:.2f} | valid {:.2f}".format(float(score1), float(score2)))

# feature importance
fi = mod_gb.feature_importances_
predictors = [x for x in df.iloc[:,4:].columns]
feat_imp = pd.Series(mod_gb.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')