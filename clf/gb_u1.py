#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:43:51 2019

@author: qianqianwang
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
import time

# read data
df_train = pd.read_csv('df37_window1_train.csv')
df_test = pd.read_csv('df37_window1_test.csv')


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
#df = df[(df["MVID"] != 47553) & (df["MVID"] != 46899) & (df["MVID"] != 67598) & (df["MVID"] != 45798)]
df = df[~df["MVID"].isin([47553,46899,67598,45798])]

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



'''7: Gradient Boosting'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
clf_gb = GradientBoostingClassifier(random_state= 1337)
clf_gb.fit(X_train, y_train)

# Predicting the train set results
y_pred = clf_gb.predict(X_train)
accuracy_score(y_train, y_pred)

# KF & GS
parameters = {'learning_rate':[0.1], 
              'max_depth':[3,4,5],
              'max_features':['auto', 'sqrt', 'log2'],
              'min_samples_leaf':[1,2,3,4],
              'min_samples_split':[2,4,6,8],
              'n_estimators':[40,50,60],
              'subsample':[0.6,0.7,0.8]}

                                       
grid_search = GridSearchCV(estimator = clf_gb,
                           param_grid = parameters,
                           scoring='accuracy',
                           cv = 5,
                           verbose = 1,
                           n_jobs = -1)

start_time = time.time()
grid_search = grid_search.fit(X_train, y_train)
print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
grid_search.best_params_, grid_search.best_score_


# last step
clf_gb = GradientBoostingClassifier(learning_rate=0.1, 
                                        max_depth = 3,
                                        max_features = 'auto',
                                        min_samples_leaf = 1,
                                        min_samples_split = 8,
                                        n_estimators=50, 
                                        subsample=0.6,
                                        random_state= 1337 )
clf_gb.fit(X_train, y_train)

# Predicting the train set results
y_pred1 = clf_gb.predict(X_train)
y_pred2 = clf_gb.predict(X_test)
score1 = accuracy_score(y_train, y_pred1) * 100
score2 = accuracy_score(y_test, y_pred2) * 100
print ("\nGB Model Report")
print("train {:.2f} | valid {:.2f}".format(float(score1), float(score2)))