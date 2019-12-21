##############################################################
#
# 500 Days of AAPL data used to predict 50 days in the future
#
# Author : Dhruv Oberoi
# Stock Price Prediction using
# Linear Regression, Ridge Regression, Lasso, SVR (RBF Kernel),
# Random Forest and XGBoost
#
####################################################################################

import quandl, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv

####################################################################################

from sklearn import preprocessing, cross_validation, svm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ARDRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from matplotlib import style
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


quandl.ApiConfig.api_key = 'PASTER YOUR API KEY HERE'

####################################################################################

style.use('ggplot')

dataset_1 = quandl.get("EOD/AAPL")
#get data from quandl website

dataset = dataset_1[['Adj_Open','Adj_High',  'Adj_Low',  'Adj_Close', 'Adj_Volume']].copy()
#dataset = dataset.iloc[pd.np.r_[:,8900:9545]]
dataset = dataset.iloc[pd.np.r_[:,-501:-1]]

dataset['HL_PCT'] = (dataset['Adj_High'] - dataset['Adj_Low']) / dataset['Adj_Low'] * 100.0
dataset['PCT_change'] = (dataset['Adj_Close'] - dataset['Adj_Open']) / dataset['Adj_Open'] * 100.0


dataset = dataset[['Adj_Close', 'HL_PCT', 'PCT_change', 'Adj_Volume']]

pred_feature = 'Adj_Close'
dataset.fillna(value=99999, inplace=True)

no_of_var = int(math.ceil(0.1 * len(dataset)))

dataset['label'] = dataset[pred_feature].shift(-no_of_var)

x = np.array(dataset.drop(['label'], 1))
x = preprocessing.scale(x)
x_small = x[-no_of_var:]
x_small = x[-no_of_var:]
x = x[:-no_of_var]

dataset.dropna(inplace=True)
y = np.array(dataset['label'])


####################################################################################

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

####################################################################################

last_date = dataset.iloc[-1].name
#name of last date
last_unix = last_date.timestamp()
one_day = 86400
#number of seconds in a day
next_unix = last_unix + one_day
#next day...

####################################################################################

dataset_2 = dataset_1[['Adj_Open','Adj_High',  'Adj_Low',  'Adj_Close', 'Adj_Volume']].copy()
#dataset_2 = dataset_2.iloc[pd.np.r_[:,9480:9545]]
dataset_2 = dataset_2.iloc[pd.np.r_[:,-51:-1]]


model1 = svm.LinearSVR()
model1.fit(x_train, y_train)
confidence1 = model1.score(x_test, y_test)
predict_1 = model1.predict(x_small)
dataset['Predict_Linear'] = np.nan
print('Score for Linear Reg: :',confidence1)
print('\n')

for i in predict_1:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

# The coefficients
print('Coefficients: \n', model1.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((model1.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model1.score(x_test, y_test))

####################################################################################

model2 = svm.SVR(C = 100, gamma = 0.1, kernel = 'rbf')

'''
param_grid = [{'C': [10, 100, 1000, 10000], 'gamma': [0.05,0.055,0.057, 0.06, 0.065,0.067,0.07, 0.075, 0.077,0.09, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9,0.95,0.97,0.98], 'kernel': ['rbf']}]
clf = GridSearchCV(model2, param_grid, cv=10)
clf.fit(x_train,y_train)
print('\n')
#print('clf =', clf)
print(clf.best_score_)
print(clf.best_params_)
print('\n')
'''

model2 = svm.SVR(kernel = 'rbf', C= 100, gamma= 0.06)
model2.fit(x_train, y_train)
confidence2 = model2.score(x_test, y_test)
predict_2 = model2.predict(x_small)
dataset['Predict_RBF'] = np.nan
print('Score for RBF Reg: :',confidence2)
print('\n')

for i in predict_2:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################
'''
model3 = svm.SVR(kernel = 'poly')

param_grid = [{'C': [10, 100, 1000], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],'degree' : [1,2,3,4,5,6,7,8,9,10], 'coef0':[0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.02],'kernel': ['poly']}]
clf = GridSearchCV(model3, param_grid, cv=10)
clf.fit(x_train,y_train)
print('\n')
#print('clf =', clf)
print(clf.best_score_)
print(clf.best_params_)
print('\n')


model3.fit(x_train, y_train)
confidence3 = model3.score(x_test, y_test)
predict_3 = model3.predict(x_small)
dataset['Predict_Poly'] = np.nan
print('Score for Poly Reg: :',confidence3)
print('\n')

for i in (predict_3):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]
'''
####################################################################################
'''
gpkernel = DotProduct() + WhiteKernel()
model4 = GaussianProcessRegressor(kernel = gpkernel, random_state = 0).fit(x_train, y_train)
confidence4 = model4.score(x_test, y_test)
predict_4 = model4.predict(x_small, return_std = True)
dataset['Predict_GPR'] = np.nan
print('Score for GPR Reg: :',confidence4)
print('\n')

for i in (predict_4):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]
'''
####################################################################################

model5 = RandomForestRegressor(n_estimators = 150, random_state = 0)
'''
param_grid = [{'n_estimators': [10,50,100,130,140,150,170,180]},
 ]
clf = GridSearchCV(model5, param_grid, cv=10)
clf.fit(x_train,y_train)
print('\n')
#print('clf =', clf)
print(clf.best_score_)
print(clf.best_params_)
print('\n')
'''

model5.fit(x_train, y_train)
confidence5 = model5.score(x_test, y_test)
predict_5 = model5.predict(x_small)
dataset['Predict_RF'] = np.nan
print('Score for RF Reg: :',confidence5)
print('\n')

for i in (predict_5):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################

model6 = XGBRegressor(max_depth=1, learning_rate=0.05, n_estimators=200, objective="reg:linear", booster='gbtree', n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, random_state=0)

'''
param_grid = [{'max_depth': [1,10,30,50,100,200,500],'n_estimators': [1,10,30,50,100,200,500]},
 ]
clf = GridSearchCV(model6, param_grid, cv=10)
clf.fit(x_train,y_train)
print('\n')
#print('clf =', clf)
print(clf.best_score_)
print(clf.best_params_)
print('\n')
'''


model6.fit(x_train, y_train)
confidence6 = model6.score(x_test, y_test)
predict_6 = model6.predict(x_small)
dataset['Predict_XGBR'] = np.nan
print('Score for XGBR Reg: :',confidence6)
print('\n')

for i in (predict_6):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################

model7 = Ridge(alpha=1, fit_intercept = True, tol = 0.001, random_state = 0, solver = 'saga')

'''
param_grid = [{'alpha': [1,10,20,30]}]
clf = GridSearchCV(model7, param_grid, cv=10)
clf.fit(x_train,y_train)
print('\n')
#print('clf =', clf)
print(clf.best_score_)
print(clf.best_params_)
print('\n')
'''

model7.fit(x_train, y_train)
confidence7 = model7.score(x_test, y_test)
predict_7 = model7.predict(x_small)
dataset['Predict_Ridge'] = np.nan
print('Score for Ridge Reg: :',confidence7)
print('\n')

for i in (predict_7):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################

model8 = Lasso(alpha=0.1,fit_intercept = True, tol = 0.01, random_state = 0, selection = 'cyclic')

'''
param_grid = [{'alpha': [0.1, 0.2, 0.5,0.07,0.05,0.03,0.01]}]
clf = GridSearchCV(model8, param_grid, cv=10)
clf.fit(x_train,y_train)
print('\n')
#print('clf =', clf)
print(clf.best_score_)
print(clf.best_params_)
print('\n')
'''

model8.fit(x_train, y_train)
confidence8 = model8.score(x_test, y_test)
predict_8 = model8.predict(x_small)
dataset['Predict_Lasso'] = np.nan
print('Score for Lasso Reg: :',confidence8)
print('\n')

for i in (predict_8):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]

####################################################################################
'''
#Bayesian Ridge Regression
model9 = ARDRegression(fit_intercept = False)
model9.fit(x_train, y_train)
confidence9 = model9.score(x_test, y_test)
predict_9 = model9.predict(x_small)
dataset['Predict_ARD'] = np.nan
print('Score for ARD Reg: :',confidence9)
print('\n')

for i in (predict_9):
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    dataset.loc[next_date] = [np.nan for _ in range(len(dataset.columns)-1)]+[i]
'''
####################################################################################


####################################################################################

dataset['Adj_Close'].plot()

dataset['Predict_Linear'].plot()
'''
#dataset['Predict_Poly'].plot()
dataset['Predict_GPR'].plot()
'''
dataset['Predict_Ridge'].plot()
dataset['Predict_Lasso'].plot()
#dataset['Predict_ARD'].plot()
dataset['Predict_RBF'].plot()
dataset['Predict_RF'].plot()
dataset['Predict_XGBR'].plot()
dataset_2['Adj_Close'].plot()
plt.legend()
plt.xticks(rotation = 45)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

####################################################################################
####################################################################################
####################################################################################

