# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 00:26:08 2019

@author: Jie.Hu
"""


''' 4: Support Vector Machine'''
from sklearn.svm import SVC
clf_SVC = SVC(random_state = 1337)
clf_SVC.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_SVC.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV
acc = cross_val_score(estimator = clf_SVC, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()

# KF n GS
parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1], 'coef0': [0,1,2,3]},
              {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1]}]

grid_search = GridSearchCV(estimator = clf_SVC,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_

# last step
clf_SVC = SVC(kernel = 'rbf',
              C = 100,
              gamma = 0.001, 
              random_state = 1337)
clf_SVC.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_SVC.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.818, 0.048
acc = cross_val_score(estimator = clf_SVC, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()



''' 6: Random Forest'''
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(random_state = 1337)
clf_RF.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_RF.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV
acc = cross_val_score(estimator = clf_RF, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()

# KF n GS
parameters = {'criterion':['gini', 'entropy'],
              'max_depth':[5,7,9],
              'max_features':[4,6,8,10],
              'min_samples_leaf':[1,4,7,10],
              'min_samples_split':[2,5,8,11],
              'n_estimators':[80, 100, 120]}
                                       
grid_search = GridSearchCV(estimator = clf_RF,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_
 
# last step
clf_RF = RandomForestClassifier(criterion='gini',
                                    max_depth = 7, 
                                    max_features = 8,
                                    min_samples_leaf = 1, 
                                    min_samples_split = 2,
                                    n_estimators=80,
                                    random_state = 1337)
clf_RF.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_RF.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.798, 0.03
acc = cross_val_score(estimator = clf_RF, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()


'''7: Gradient Boosting'''
from sklearn.ensemble import GradientBoostingClassifier
clf_GB = GradientBoostingClassifier(random_state= 1337)
clf_GB.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_GB.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV
acc = cross_val_score(estimator = clf_GB, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()

parameters = {'learning_rate':[0.03, 0.05, 0.07], 
              'max_depth':[3,4,5],
              'max_features':[3,4,5],
              'min_samples_leaf':[3,4,5],
              'min_samples_split':[2,3,4],
              'n_estimators':[60,70,80], 
              'subsample':[0.6, 0.7,0.8]}
                                       
grid_search = GridSearchCV(estimator = clf_GB,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_

# last step
#(0.05, 6, 3, 2, 3, 80, 0.7, 1337)
#(0.05, 4, 4, 4, 2, 70, 0.7, 1337)
clf_GB = GradientBoostingClassifier(learning_rate=0.05, 
                                        max_depth = 6,
                                        max_features = 3,
                                        min_samples_leaf = 2,
                                        min_samples_split = 4,
                                        n_estimators=80, 
                                        subsample=0.7,
                                        random_state= 1337 )
clf_GB.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_GB.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.814, 0.019
acc = cross_val_score(estimator = clf_GB, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()




''' gradient boosting '''
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
mod_gb = GradientBoostingRegressor(random_state= 1337)
mod_gb.fit(X, y)

# Predicting the Test set results
y_pred = mod_gb.predict(X)
mape_gb = np.mean(np.abs((np.expm1(y) - np.expm1(y_pred)) / np.expm1(y))) * 100
#Print model report:
print "\nModel Report"
print "MAPE : %.2f" % mape_gb



parameters = {'learning_rate':[0.04, 0.05, 0.06], 
              'max_depth':[7,8,9],
              'max_features':[4,5,6],
              'min_samples_leaf':[2,3,4],
              'min_samples_split':[2,3,4],
              'n_estimators':[80,100,120],
              'subsample':[0.8,0.9]}
                                       
grid_search = GridSearchCV(estimator = mod_gb,
                           param_grid = parameters,
                           scoring='neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_

# last step
mod_gb = GradientBoostingRegressor(learning_rate=0.04, 
                                        max_depth = 9, 
                                        max_features = 4,
                                        min_samples_leaf = 4, 
                                        min_samples_split = 2,
                                        n_estimators=100, 
                                        subsample=0.8,
                                        random_state= 1337 )
mod_gb.fit(X, y)

# Predicting the Test set results
y_pred = mod_gb.predict(X)

# check r2 and mape
mape_gb = np.mean(np.abs((np.expm1(y) - np.expm1(y_pred)) / np.expm1(y))) * 100
r2_gb = r2_score(np.expm1(y), np.expm1(y_pred))

#Print model report:
print "\nModel Report"
print "R2 Score: %.2f" % r2_gb
print "MAPE Score: %.2f" % mape_gb

# feature importance
fi = mod_gb.feature_importances_
predictors = [x for x in bo.columns if x not in ['Date', 'Title', 'OBO']]
feat_imp = pd.Series(mod_gb.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')