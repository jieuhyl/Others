#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:17:14 2019

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
#from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR


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
print ("\nxgb Model Report")
print ("MAPE : %.2f" % mape)


# lgb =========================================================================
mod_lgb = lgb.LGBMRegressor(random_state=1337)
mod_lgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = mod_lgb.predict(X_train)
mape = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred)) / np.expm1(y_train))) * 100
#Print model report:
print ("\nlgb Model Report")
print ("MAPE : %.2f" % mape)