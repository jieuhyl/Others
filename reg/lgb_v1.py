#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:42:05 2019

@author: qianqianwang
"""

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
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR

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

# lgb =========================================================================
mod_lgb = lgb.LGBMRegressor(random_state=1337)
mod_lgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = mod_lgb.predict(X_train)
mape = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred)) / np.expm1(y_train))) * 100
#Print model report:
print ("\nLGB Model Report")
print ("MAPE : %.2f" % mape)

'''
def run_lgb(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    params = {
        "objective" : 'regression',
        "metric" : 'rmse',
        "boosting": 'gbdt',
        "learning_rate" : 0.1,
        "num_leaves" : 40, 
        "min_sum_hessian_in_leaf" : 0.1,
        "bagging_freq" : 3,
        "bagging_fraction" : 0.6,
        "feature_fraction" : 0.6,
        "lambda_l1" : 0.1,
        "lambda_l2" : 0.1,
        "verbosity" : -1,
        "random_state": 1337
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)
    evals_result = {}
    mod_lgb = lgb.train(params, 
                      dtrain, 
                      valid_sets=[dtrain, dvalid], 
                      num_boost_round = 500, 
                      early_stopping_rounds=50, 
                      verbose_eval=100, 
                      evals_result=evals_result)
    
    y_pred = mod_lgb.predict(X_test, num_iteration=mod_lgb.best_iteration)
    score = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred)) / np.expm1(y_test))) * 100
    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return score, mod_lgb, y_pred, evals_result

run_lgb(X_train, X_test, y_train, y_test)
'''

start_time = time.time()

def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

params = {
    "objective" : 'regression',
    "metric" : 'mape',
    "boosting": 'gbdt',
    "learning_rate" : 0.1,
    "num_leaves" : 30, 
    "min_sum_hessian_in_leaf" : 0,
    "bagging_freq" : 3,
    "bagging_fraction" : 0.6,
    "feature_fraction" : 0.6,
    "lambda_l1" : 0.01,
    "lambda_l2" : 0.1,
    "verbosity" : -1,
    "random_state": 1337
    }

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_test, label=y_test)
evals_result = {}
mod_lgb = lgb.train(params, 
                  dtrain, 
                  valid_sets=[dtrain, dvalid], 
                  num_boost_round = 500, 
                  early_stopping_rounds=50, 
                  verbose_eval=100, 
                  feature_name=df.iloc[:,4:].columns.tolist(),
                  callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
                  evals_result=evals_result)

y_pred1 = mod_lgb.predict(X_train, num_iteration=mod_lgb.best_iteration)
y_pred2 = mod_lgb.predict(X_test, num_iteration=mod_lgb.best_iteration)
score1 = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred1)) / np.expm1(y_train))) * 100
score2 = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred2)) / np.expm1(y_test))) * 100
print ("\nLGB Model Report")
print("train {:.2f} | valid {:.2f}".format(float(score1), float(score2)))


# plot
print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='mape')
plt.show()

print('Plot feature importances...')
ax = lgb.plot_importance(mod_lgb, max_num_features=10)
plt.show()

print('Plotting split value histogram...')
ax = lgb.plot_split_value_histogram(mod_lgb, feature='CRITICSCORE', bins='auto')
plt.show()


# hyperparameters tuning
def my_scorer(y_true, y_pred):
    mape = np.mean(np.abs((np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true))) * 100
    return mape
my_func = make_scorer(my_scorer, greater_is_better=False)


cv_lgb = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=300, objective='regression', verbosity = -1, random_state= 1337)

parameters = {
              'learning_rate':[0.1],
              'num_leaves':[5,10,15,20], 
              'min_sum_hessian_in_leaf':[0],
              'bagging_freq' : [0,3,5],
              'bagging_fraction':[0.6],
              'feature_fraction':[0.6],
              'lambda_l1':[0.01,0.1,0,1],
              'lambda_l2':[0.01,0.1,0,1]
              
              }

grid_search = GridSearchCV(estimator = cv_lgb,
                           param_grid = parameters,
                           #scoring='neg_mean_squared_error',
                           scoring = my_func,
                           cv = 5,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# feature importance
attr2 = {k: v for k, v in zip(df.iloc[:,4:].columns, mod_lgb.feature_importance) if v>0}
attr2 = sorted(attr2.items(), key=lambda x: x[1], reverse = False)
x1,y1 = zip(*attr2)
i1=range(len(x1))
plt.figure(num=None, figsize=(9, 7))
plt.barh(i1, y1)
plt.title("LGBM")
plt.yticks(i1, x1)
plt.show();


print("Features Importance...")
gain = mod_lgb.feature_importance(importance_type='gain')
featureimp = pd.DataFrame({'feature':df.iloc[:,4:].columns, 
                   'split':mod_lgb.feature_importance(importance_type='split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:10])
   
    
# Fit model using each importance as a threshold
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.feature_selection import SelectFromModel

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

thresholds = sorted(model.feature_importances_, reverse= True)
for thresh in thresholds[:20]:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = lgb.LGBMRegressor()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	#predictions = [round(value) for value in y_pred]
	accuracy = explained_variance_score(y_test, y_pred)
	print("Thresh=%.3f, n=%d, R^2: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))