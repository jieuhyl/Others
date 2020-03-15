# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 15:08:52 2019

@author: Jie.Hu
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


''' modeling '''
''' Ensemble '''
# define cross validation strategy
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

'''
models = [LinearRegression(),
          make_pipeline(RobustScaler(), Ridge(alpha=42)),
          make_pipeline(RobustScaler(), Lasso(alpha=0.0057, max_iter=10000)),
          make_pipeline(RobustScaler(), ElasticNet(alpha=0.011, l1_ratio=0.47, max_iter=10000)),
          make_pipeline(RobustScaler(), SVR(kernel = 'rbf', C = 5, gamma = 0.001)),
          RandomForestRegressor(max_depth = 10, max_features = 40, n_estimators=600, random_state=1337),
          GradientBoostingRegressor(learning_rate=0.01, n_estimators=1200, subsample=0.2, random_state=1337)]

names = ["LR", "Ridge", "Lasso", "ENet", "SVR", "RF", "GB"]
for name, model in zip(names, models):
    score = rmse_cv(model, X_train, y_train)
    print("{}: {:.3}, {:.3f}".format(name,score.mean(),score.std()))


# xgboost
mod_xgb = XGBRegressor(random_state= 1337)
mod_xgb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = mod_xgb.predict(X_train)
mape = np.mean(np.abs((np.expm1(y_train) - np.expm1(y_pred)) / np.expm1(y_train))) * 100
#Print model report:
print "\nModel Report"
print "MAPE : %.2f" % mape



parameters = {'colsample_bytree':[0.3,0.5,0.7],
              'gamma':[0.01,0.1], 
              'learning_rate':[0.01,0.1,0.2],
              'max_depth':[5,10,15], 
              'min_child_weight':[1,2,3],
              'n_estimators':[100,200,400],
              'reg_alpha':[0.1,0.3,0.5,0.7],
              'reg_lambda':[0.1,0.3,0.5,0.7],
              'subsample':[0.2,0.4,0.6,0.8]}

parameters = {'gamma':[0.1], 
              'learning_rate':[0.1],
              'max_depth':[10], 
              'min_child_weight':[1],
              'n_estimators':[1000],
              'subsample':[0.6]}

parameters = {'gamma':[0.1], 
              'learning_rate':[0.01],
              'max_depth':[3,4], 
              'min_child_weight':[1,2],
              'n_estimators':[4000,5000,6000],
              'reg_alpha':[0.01],
              'reg_lambda':[0.1],
              'subsample':[0.1,0.2,0.3]}

grid_search = GridSearchCV(estimator = mod_xgb,
                           param_grid = parameters,
                           scoring='neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_
'''
#==============================================================================
# Average base models according to their weights.
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w


mod_ridge = make_pipeline(RobustScaler(), Ridge(alpha=42))
mod_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0057, max_iter=10000))
mod_enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.011, l1_ratio=0.47, max_iter=10000))
mod_svr = make_pipeline(RobustScaler(), SVR(kernel = 'rbf', C = 5, gamma = 0.001))
mod_gb = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1200, subsample=0.2, random_state=1337)
mod_xg = XGBRegressor(gamma = 0.1, learning_rate = 0.01, max_depth = 3, min_child_weight = 1, n_estimators = 4000, reg_alpha = 0.01, reg_lambda = 0.1, subsample = 0.1, random_state= 1337)

weights = [0.2,0.2,0.6]

mod1 = AverageWeight(mod = [mod_ridge, mod_lasso, mod_enet], weight=weights)

score = rmse_cv(mod1,X_train,y_train)
print(score.mean())

mod1.fit(X_train, y_train)
stacked_train_pred = mod1.predict(X_train)
print(rmse(y_train, stacked_train_pred))

# Average base models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
    
mod2 = AveragingModels(models = (mod_ridge, mod_lasso, mod_enet, mod_svr))

score = rmse_cv(mod2, X_train, y_train)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

mod2.fit(X_train, y_train)
stacked_train_pred = mod2.predict(X_train)
print(rmse(y_train, stacked_train_pred))
    
# Stacking models
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

    
mod3 = StackingAveragedModels(base_models = [mod_ridge, mod_lasso, mod_enet],
                              meta_model = mod_svr)

score = rmse_cv(mod3, X_train, y_train) 
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

mod3.fit(X_train, y_train)
stacked_train_pred = mod3.predict(X_train)
print(rmse(y_train, stacked_train_pred))

stacked_test_pred = mod3.predict(X_test)
print(rmse(y_test, stacked_test_pred))



# MAPE
np.mean(np.abs(np.expm1(stacked_train_pred) - np.expm1(y_train)))/np.mean(np.expm1(y_train))

np.mean(np.abs(np.expm1(stacked_test_pred) - np.expm1(y_test)))/np.mean(np.expm1(y_test))



# TEST ========================================================================

df_test = pd.read_csv('test_endgame.csv')

# check missing
df_test.isnull().values.sum()

for feat in skewed_features:
    #all_data[feat] += 1
    #df[feat] = boxcox1p(df[feat], lam)
    df_test[feat] = np.log1p(df_test[feat])
    
# get mtx
X_test = df_test.iloc[:, 4:].values

pred1 = np.expm1(mod1.predict(X_test))/1000000
pred2 = np.expm1(mod2.predict(X_test))/1000000
pred3 = np.expm1(mod3.predict(X_test))/1000000

print("forecast: mod1 {:.1f} | mod2 {:.1f} | mod3 {:.1f}".format(float(pred1), float(pred2), float(pred3)))