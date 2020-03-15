# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 18:09:19 2018

@author: Jie.Hu

Building a stacking classification model for window 0.5W
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings('ignore') 
from scipy import stats
from scipy.stats import norm, skew 

# read data
df_train = pd.read_csv('df37_window1_train.csv')
df_test = pd.read_csv('df37_window1_test.csv')

# check outliers
# UA
df_train['UA_0.5W_T'].hist(bins = 50)
plt.scatter(df_train['UA_0.5W_T'], df_train['OBO'])
# TA
df_train['TA_0.5W_T'].hist(bins = 50)
plt.scatter(df_train['TA_0.5W_T'], df_train['OBO'])
# DI
df_train['DI_0.5W_T'].hist(bins = 50)
plt.scatter(df_train['DI_0.5W_T'], df_train['OBO'])
# FC
df_train['FC_0.5W_T'].hist(bins = 50)
plt.scatter(df_train['FC_0.5W_T'], df_train['OBO'])


from collections import Counter

# Outlier detection 
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(df_train, 2, ['UA_0.5W_T', 'TA_0.5W_T', 'DI_0.5W_T', 'FC_0.5W_T'])

# show the outliers
df_train_drop = df_train.loc[Outliers_to_drop] 

# drop the outliers
#df = df.drop(Outliers_to_drop, axis = 0, inplace = True)

# concatenate train and test
ntrain = df_train.shape[0]
ntest = df_test.shape[0]

df = pd.concat([df_train, df_test]).reset_index(drop=True)
# all_data = pd.concat((train, test)).reset_index(drop=True)

'''
Mapping all the dataframe
def mapping(x):
    if x == 5:
        val = 1
    if x != 5:
        val = 0
    return val
    
df2 = df1.applymap(mapping)
'''
# add Cate variable
# create a group 
def f(row):
    if row['OBO'] <= 10:
        val = "D"
    elif row['OBO'] > 10 and row['OBO'] <= 50:
        val = "C"
    elif row['OBO'] > 50 and row['OBO'] <= 100:
        val = "B"
    else:
        val = "A"
    return val

df['CATEGORY'] = df.apply(f, axis=1)


# check missing
df.isnull().values.sum()
missing_ratio = df.isnull().sum() / len(df)
missing_ratio.sort_values(ascending=True)[10:]

# check UA_T
null_data = df[df['UA_0.5W_T'].isnull()]

# drop four 
df = df[(df["MVID"] != 47553) & (df["MVID"] != 46899) & (df["MVID"] != 67598) & (df["MVID"] != 45798)]

# check FCO
null_data = df[df['FCO_0.5W_T'].isnull()]

# fil missing values with group median
sns.boxplot(x="CATEGORY", y = "FCO_0.5W_T", data=df)

# get median
df[['FCO_0.5W_T', 'FCO_0.5W_M24', 'FCO_0.5W_M26', 'FCO_0.5W_F24', 'FCO_0.5W_F26', 'CATEGORY']].groupby('CATEGORY').median()

# FCO_0.5W_T
def impute_FCO(cols):
    FCO = cols[0]
    GROUP = cols[1]
    
    if pd.isnull(FCO):

        if GROUP == "A":
            return 42.5
        elif GROUP == "B":
            return 25    
        elif GROUP == "C":
            return 14
        else:
            return 7

    else:
        return FCO
df['FCO_0.5W_T'] = df[['FCO_0.5W_T','CATEGORY']].apply(impute_FCO, axis=1)

# FCO_0.5W_M24
def impute_FCO1(cols):
    FCO = cols[0]
    GROUP = cols[1]
    
    if pd.isnull(FCO):

        if GROUP == "A":
            return 49
        elif GROUP == "B":
            return 25    
        elif GROUP == "C":
            return 14
        else:
            return 6

    else:
        return FCO
df['FCO_0.5W_M24'] = df[['FCO_0.5W_M24','CATEGORY']].apply(impute_FCO1, axis=1)

# FCO_0.5_M26
def impute_FCO2(cols):
    FCO = cols[0]
    GROUP = cols[1]
    
    if pd.isnull(FCO):

        if GROUP == "A":
            return 45
        elif GROUP == "B":
            return 28   
        elif GROUP == "C":
            return 13
        else:
            return 7

    else:
        return FCO
df['FCO_0.5W_M26'] = df[['FCO_0.5W_M26','CATEGORY']].apply(impute_FCO2, axis=1)

# FCO_0.5_F24
def impute_FCO3(cols):
    FCO = cols[0]
    GROUP = cols[1]
    
    if pd.isnull(FCO):

        if GROUP == "A":
            return 41.5
        elif GROUP == "B":
            return 24    
        elif GROUP == "C":
            return 14
        else:
            return 6

    else:
        return FCO
df['FCO_0.5W_F24'] = df[['FCO_0.5W_F24','CATEGORY']].apply(impute_FCO, axis=1)

# FCO_0.5W_F26
def impute_FCO4(cols):
    FCO = cols[0]
    GROUP = cols[1]
    
    if pd.isnull(FCO):

        if GROUP == "A":
            return 37
        elif GROUP == "B":
            return 26    
        elif GROUP == "C":
            return 14
        else:
            return 8

    else:
        return FCO
df['FCO_0.5W_F26'] = df[['FCO_0.5W_F26','CATEGORY']].apply(impute_FCO4, axis=1)


# check missing again
df.isnull().values.sum()
missing_ratio = df.isnull().sum() / len(df)
missing_ratio.sort_values(ascending=False)

# fill HOLIDAY with normal
df['HOLIDAY'] = df['HOLIDAY'].fillna("NORMAL")

# drop features
drop_feats = ['WW_GRS', 
              'PERCENT', 
              'NM_0.5W_T', 
              'NM_0.5W_M24', 
              'NM_0.5W_M26',
              'NM_0.5W_F24', 
              'NM_0.5W_F26', 
              'GENRE2']

df.drop(drop_feats, axis = 1, inplace = True)


# check OBO
df['OBO'].describe()

# orginal data
sns.distplot(df['OBO'] , fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df['OBO'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df['OBO'], plot=plt)
plt.show()

# log transformation
sns.distplot(np.log(df['OBO']), fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(np.log(df['OBO']))
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(np.log(df['OBO']), plot=plt)
plt.show()


# GENRE1
df['GENRE1'].value_counts()

# graph
g = sns.boxplot(x = "GENRE1", y = "OBO", data = df)
plt.setp(g.get_xticklabels(), rotation=90)

# RATING
df['RATING'].value_counts(ascending = False)
# graph
g = sns.boxplot(x = "RATING", y = "OBO", data = df)

# STUDIO
df['STUDIO'].value_counts(ascending = False)
# graph
g = sns.boxplot(x = "STUDIO", y = "OBO", data = df)

# divide studio into 4 parts
Studio_counts = df['STUDIO'].value_counts(ascending = False).tolist()
pd.qcut(Studio_counts, 4)

# create studio counts
df['STUDIO_COUNTS'] = df.groupby(['STUDIO'])['MVID'].transform('count')

# create a group 
def f(row):
    if row['STUDIO_COUNTS'] > 18:
        val = "SUPER"
    elif row['STUDIO_COUNTS'] > 5 and row['STUDIO_COUNTS'] <= 18:
        val = "BIG"
    elif row['STUDIO_COUNTS'] > 2 and row['STUDIO_COUNTS'] <= 5:
        val = "MEDIUM"
    else:
        val = "SMALL"
    return val

df['STUDIO_GROUP'] = df.apply(f, axis=1)

# MONTH
month = {1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5:'MAY', 6:'JUN', 
         7:'JUL', 8:'AUG', 9:'SEP', 10:'OCT', 11:'NOV', 12:'DEC'}
df['MONTH'] = df['MONTH'].map(month)
# graph
g = sns.boxplot(x = "MONTH", y = "OBO", data = df,
                order=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                       'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])

# further drop some features
drop_feats2 = ['WD', 'MVID', 'STUDIO', 'DATE', 'WKNDDAY', 'WKNM', 'WKNDGRS', 
               'WKGRS', 'PER_SCN', 'RANK', 'GRS', 'STUDIO_COUNTS']

df.drop(drop_feats2, axis = 1, inplace = True)

# check numerical features
# Screens
df['SCRNS'].hist(bins = 75)
plt.scatter(df['SCRNS'], df['OBO'])

#correlation matrix
corrmat = df.corr()
sns.heatmap(corrmat,  cmap="YlGnBu", square=True)

# numerical skewness
numeric_feats = df.dtypes[df.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[np.abs(skewness) >= 0.75]
#print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    df[feat] = boxcox1p(df[feat], lam)
    
    
# categorical dummmy
# GENRE1
t_dummies1  = pd.get_dummies(df['GENRE1'], prefix='GENRE1')
t_dummies1.drop(['GENRE1_DOCUMENTARY'], axis=1, inplace=True)
df = df.join(t_dummies1)
df.drop(['GENRE1'], axis=1, inplace=True)  

# RATING
t_dummies2 = pd.get_dummies(df['RATING'], prefix='RATING')
t_dummies2.drop(['RATING_G'], axis=1, inplace=True)
df = df.join(t_dummies2)
df.drop(['RATING'], axis=1, inplace=True) 

# MONTH
t_dummies3 = pd.get_dummies(df['MONTH'], prefix='MONTH')
t_dummies3.drop(['MONTH_FEB'], axis=1, inplace=True)
df = df.join(t_dummies3)
df.drop(['MONTH'], axis=1, inplace=True) 

# HOLIDAY
t_dummies4 = pd.get_dummies(df['HOLIDAY'], prefix='HOLIDAY')
t_dummies4.drop(['HOLIDAY_NEW YEAR'], axis=1, inplace=True)
df = df.join(t_dummies4)
df.drop(['HOLIDAY'], axis=1, inplace=True) 

# STUDIO_GROUP
t_dummies5 = pd.get_dummies(df['STUDIO_GROUP'], prefix='STUDIO_GROUP')
t_dummies5.drop(['STUDIO_GROUP_SMALL'], axis=1, inplace=True)
df = df.join(t_dummies5)
df.drop(['STUDIO_GROUP'], axis=1, inplace=True) 


# last check missing
df.isnull().values.sum()


# get the train, test and mtx
train = df[:ntrain]
test = df[ntrain:]

#
train_name = train['MVNAME']
test_name = test['MVNAME']

y_train = train['CATEGORY'].values
y_test = test['CATEGORY'].values

drop_feats3 = ['MVNAME', 'OBO', 'Year', 'CATEGORY']

train.drop(drop_feats3, axis = 1, inplace = True)
test.drop(drop_feats3, axis = 1, inplace = True)

X_train = train.values
X_test = test.values


''' MODELING'''
''' 1: Logistic Regression '''
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV 
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(random_state = 1337)
clf_LR.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_LR.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.766, 0.044
acc = cross_val_score(estimator = clf_LR, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()


''' 2: Naive Bayes'''
from sklearn.naive_bayes import MultinomialNB
clf_NB = MultinomialNB()
clf_NB.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_NB.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.693, 0.047
acc = cross_val_score(estimator = clf_NB, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()


''' 3: K Nearest Neighbor'''
from sklearn.neighbors import KNeighborsClassifier
clf_KNN = KNeighborsClassifier()
clf_KNN.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_KNN.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV
acc = cross_val_score(estimator = clf_KNN, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()

# KF n GS
parameters = {'n_neighbors': [5, 6, 7, 8, 9, 10], 
              'metric': ['minkowski', 'euclidean', 'manhattan'], 
              'weights': ['uniform', 'distance']}
                                       
grid_search = GridSearchCV(estimator = clf_KNN,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_

# last step
clf_KNN = KNeighborsClassifier(metric = 'minkowski',
                               n_neighbors = 9,
                               weights = 'distance')
clf_KNN.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_KNN.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.781, 0.044
acc = cross_val_score(estimator = clf_KNN, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()


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


''' 5: Decision Tree'''
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
clf_DT = DecisionTreeClassifier(random_state = 1337)
clf_DT.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_DT.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV
acc = cross_val_score(estimator = clf_DT, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()



parameters = {'criterion':['gini', 'entropy'],
              'max_depth':[3,4,5,6,7,8],
              'max_features':[4,5,6,7,8,9,10],
              'min_samples_leaf':[1,2,3,4,5],
              'min_samples_split':[2,3,4,5,6,7,8,9]}
                                       
grid_search = GridSearchCV(estimator = clf_DT,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
grid_search.best_params_ , grid_search.best_score_


# last step
clf_DT = DecisionTreeClassifier(criterion = 'gini',
                                max_depth = 6, 
                                max_features = 10,
                                min_samples_leaf = 3,
                                min_samples_split = 2,
                                random_state= 1337 )
clf_DT.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_DT.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.733, 0.048
acc = cross_val_score(estimator = clf_DT, X = X_train, y = y_train, cv = 10)
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


'''1: Ensemble '''
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
clf_LR = LogisticRegression(random_state = 1337)

clf_NB = MultinomialNB()


clf_KNN = KNeighborsClassifier(metric = 'minkowski',
                               n_neighbors = 9,
                               weights = 'distance')

clf_SVC = SVC(kernel = 'rbf',
              C = 100,
              gamma = 0.001, 
              probability=True,
              random_state = 1337)

clf_DT = DecisionTreeClassifier(criterion = 'gini',
                                max_depth = 6, 
                                max_features = 10,
                                min_samples_leaf = 3,
                                min_samples_split = 2,
                                random_state= 1337 )

clf_RF = RandomForestClassifier(criterion='gini',
                                    max_depth = 7, 
                                    max_features = 8,
                                    min_samples_leaf = 1, 
                                    min_samples_split = 2,
                                    n_estimators=80,
                                    random_state = 1337)

clf_GB = GradientBoostingClassifier(learning_rate=0.05, 
                                        max_depth = 6,
                                        max_features = 3,
                                        min_samples_leaf = 2,
                                        min_samples_split = 4,
                                        n_estimators=80, 
                                        subsample=0.7,
                                        random_state= 1337 )

clf_ensemble = VotingClassifier(estimators=[('LR', clf_LR), 
                                            ('NB', clf_NB), 
                                            ('KNN', clf_KNN),
                                            ('SVC', clf_SVC),
                                            ('DT', clf_DT),
                                            ('RF', clf_RF),
                                            ('GB', clf_GB)], 
                                            weights=[0,0,0,3,0,1,2],
                                            voting='soft')

clf_ensemble.fit(X_train, y_train)
y_pred = clf_ensemble.predict(X_train)
metrics.accuracy_score(y_train, y_pred)

# CV  0.819  0.03
acc = cross_val_score(estimator = clf_ensemble, X = X_train, y = y_train, cv = 10)
acc.mean(), acc.std()

'''
2: Ensemble 
from sklearn.cross_validation import KFold
# Some useful parameters which will come in handy later on
# ntrain = df_train.shape[0]
# ntest = df_test.shape[0]
SEED = 1337 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    



def get_oof(clf, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntrain,))
    oof_test_skf = np.empty((NFOLDS, ntrain))

    for i, (train_index, test_index) in enumerate(kf):
        X_tr = X_train[train_index]
        y_tr = y_train[train_index]
        X_te = X_train[test_index]

        clf.train(X_tr, y_tr)

        oof_train[test_index] = clf.predict(X_te)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Parameters
# LR
lr_params = {}

# SVC
svc_params = {'kernel': 'rbf',
              'C': 100,
              'gamma': 0.001, 
              'probability': 'True'}

# DT
dt_params = {'criterion': 'gini',
             'max_depth': 6, 
             'max_features': 10,
             'min_samples_leaf': 3, 
             'min_samples_split': 2,
             'n_estimators': 80}

# RF
rf_params = {'criterion': 'gini',
             'max_depth': 7, 
             'max_features': 8,
             'min_samples_leaf': 1, 
             'min_samples_split': 2,
             'n_estimators': 80}

#GB
gb_params = {'learning_rate': 0.05, 
             'max_depth': 6,
             'max_features': 3,
             'min_samples_leaf': 2,
             'min_samples_split': 4,
             'n_estimators': 80, 
             'subsample': 0.7}


lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)
dt = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=dt_params)
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)


# Create our OOF train and test predictions. These base results will be used as new features
lr_oof_train, lr_oof_test = get_oof(lr, X_train, y_train, X_train) 
svc_oof_train, svc_oof_test = get_oof(svc, X_train, y_train, X_train) 
dt_oof_train, dt_oof_test = get_oof(dt, X_train, y_train, X_train)  
rf_oof_train, rf_oof_test = get_oof(rf, X_train, y_train, X_train) 
gb_oof_train, gb_oof_test = get_oof(gb,X_train, y_train, X_train) 

'''

''' 2: Ensembel'''
#1. get LR train
clf_LR.fit(X_train, y_train)
train_LR = clf_LR.predict(X_train)
clf_NB.fit(X_train, y_train)
train_NB = clf_NB.predict(X_train)
clf_KNN.fit(X_train, y_train)
train_KNN = clf_KNN.predict(X_train)
clf_SVC.fit(X_train, y_train)
train_SVC = clf_SVC.predict(X_train)
clf_DT.fit(X_train, y_train)
train_DT = clf_DT.predict(X_train)
clf_RF.fit(X_train, y_train)
train_RF = clf_RF.predict(X_train)
clf_GB.fit(X_train, y_train)
train_GB = clf_GB.predict(X_train)

train_all = pd.DataFrame({'NAME': train_name,
                          'LR': train_LR,
                          'NB': train_NB,
                          'KNN': train_KNN,
                          'SVC': train_SVC,
                          'DT': train_DT,
                          'RF': train_RF,
                          'GB': train_GB,
                          'RESULT': y_train})

train_all.drop(['NAME'], axis = 1, inplace = True)


def mapping(x):
    if x == 'D':
        val = 1
    elif x == 'C':
        val = 2
    elif x == 'B':
        val = 3
    else:
        val = 4
    return val
    
train_all = train_all.applymap(mapping)

corr = train_all.corr()
sns.heatmap(corr, cmap = 'viridis', annot=True, linewidths = 0.5,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

train_all.drop(['RESULT'], axis = 1, inplace = True)


X_all = train_all.values
y_all = np.array(map(mapping, y_train))


''' xgboost '''
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

clf_XGB = XGBClassifier(random_state= 1337)
clf_XGB.fit(X_all, y_all)

# Predicting the train set results
y_pred = clf_XGB.predict(X_all)
metrics.accuracy_score(y_all, y_pred)

# CV
acc = cross_val_score(estimator = clf_XGB, X = X_all, y = y_all, cv = 10)
acc.mean(), acc.std()


parameters = {'colsample_bytree':[0.3,0.5,0.7,0.9],
              'gamma':[0.001, 0.005, 0.01], 
              'learning_rate':[0.01,0.05,0.1],
              'max_depth':[3,5,7,9], 
              'min_child_weight':[1,3,5,7,9],
              'n_estimators':[80,100,120],
              'subsample':[0.3,0.5,0.7,0.9]}

grid_search = GridSearchCV(estimator = clf_XGB,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_all, y_all)
grid_search.best_params_, grid_search.best_score_

# last step
clf_XGB = XGBClassifier(colsample_bytree=0.3,
                        gamma=0.001,
                        learning_rate=0.01, 
                        max_depth=3,
                        min_child_weight=1, 
                        n_estimators=80, 
                        subsample=0.5,
                        random_state= 1337)
clf_XGB.fit(X_all, y_all)

# Predicting the train set results
y_pred = clf_XGB.predict(X_all)
metrics.accuracy_score(y_all, y_pred)

# CV  0.998， 0.006
acc = cross_val_score(estimator = clf_XGB, X = X_all, y = y_all, cv = 10)
acc.mean(), acc.std()

look_all = pd.DataFrame({'NAME': train_name,
                          'LR': train_LR,
                          'NB': train_NB,
                          'KNN': train_KNN,
                          'SVC': train_SVC,
                          'DT': train_DT,
                          'RF': train_RF,
                          'GB': train_GB,
                          'RESULT': y_train,
                          'RESULT_NUM': y_all,
                          'ENSEMBLE': y_pred})


# fit for the test data
clf_LR.fit(X_train, y_train)
test_LR = clf_LR.predict(X_test)
clf_NB.fit(X_train, y_train)
test_NB = clf_NB.predict(X_test)
clf_KNN.fit(X_train, y_train)
test_KNN = clf_KNN.predict(X_test)
clf_SVC.fit(X_train, y_train)
test_SVC = clf_SVC.predict(X_test)
clf_DT.fit(X_train, y_train)
test_DT = clf_DT.predict(X_test)
clf_RF.fit(X_train, y_train)
test_RF = clf_RF.predict(X_test)
clf_GB.fit(X_train, y_train)
test_GB = clf_GB.predict(X_test)

test_all = pd.DataFrame({ 'LR': test_LR,
                          'NB': test_NB,
                          'KNN': test_KNN,
                          'SVC': test_SVC,
                          'DT': test_DT,
                          'RF': test_RF,
                          'GB': test_GB})

def mapping(x):
    if x == 'D':
        val = 1
    elif x == 'C':
        val = 2
    elif x == 'B':
        val = 3
    else:
        val = 4
    return val
    
test_all = test_all.applymap(mapping)

X_test_final = test_all.values
y_test_final = np.array(map(mapping, y_test))

# Predicting the train set results
y_pred = clf_XGB.predict(X_test_final)
metrics.accuracy_score(y_test_final, y_pred)

# CV
acc = cross_val_score(estimator = clf_XGB, X = X_test_final, y = y_test_final, cv = 10)
acc.mean(), acc.std()

look_all = pd.DataFrame({'NAME': test_name,
                          'LR': test_LR,
                          'NB': test_NB,
                          'KNN': test_KNN,
                          'SVC': test_SVC,
                          'DT': test_DT,
                          'RF': test_RF,
                          'GB': test_GB,
                          'RESULT': y_test,
                          'ENSEMBLE': y_pred})
