import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,r2_score,mean_squared_error
from sklearn.datasets import load_iris
data = pd.read_csv('HousingData.csv')
# print(data)
# Logistic Regression...
data['highest_price'] = (data['MEDV']>25).astype(int)
print(data['highest_price'])

'''Manual formula for Logistic Regression''' 
# x = data['RM']
# y = data['highest_price']
# z= 7
# x_mean = data['RM'].mean()
# y_mean = data['highest_price'].mean()
# numerator = ((x-x_mean)*(y-y_mean)).sum()
# denominator = ((x-x_mean)**2).sum()
# slope = numerator/denominator
# print(slope)
# intercept = y_mean-(slope*x_mean)
# print(intercept)
# power_e = -(intercept + (slope*z))
# print(power_e)
# LogisticRegression_formula = 1/(1+math.exp(power_e))
# print(LogisticRegression_formula)
# x = data[['RM']]
# y = data['highest_price']
# model = LogisticRegression()
# model.fit(x,y)
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f'slope: {slope} and intercept: {intercept}')
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# accuracy_score_test = accuracy_score(y_test,y_pred)
# confusion_matrix_test = confusion_matrix(y_test,y_pred)
# classification_report_test = classification_report(y_test,y_pred)
# print(accuracy_score_test)
# print(f"classify:{classification_report_test}")
# print(confusion_matrix_test)
## '''another Problem '''
print(data)
data['LSTAT_New'] = data['LSTAT'].fillna(data['LSTAT'].mean())
print(data['LSTAT_New'])
data['RM_New'] = (data['RM']>7).astype(int)
print(data['RM_New'])
print(data.describe())
print((data.isnull().sum()*100/len(data)))
print(data.isna().sum())
print(data['LSTAT'].value_counts())
a = data[['LSTAT_New']]
b = data['RM_New']
model= LogisticRegression()
print(model)
model.fit(a,b)
slope_1 = model.coef_[0]
intercept_1 = model.intercept_
print(f"slope: {slope_1} & intercept: {intercept_1}")
x_train,x_test,y_train,y_test = train_test_split(a,b,test_size=0.3,random_state=42)
model.fit(x_train,y_train)
model_pred = model.predict(x_test)
accuracy_score_test = accuracy_score(y_test,model_pred)
confusion_matrix_test = confusion_matrix(y_test,model_pred)
classification_report_test = classification_report(y_test,model_pred)
print(accuracy_score_test)
print(confusion_matrix_test)
print(classification_report_test)