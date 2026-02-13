import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.metrics import r2_score,mean_squared_error
data_set = pd.read_csv("Boston.csv")
# print(data_set)
num_columns = data_set.select_dtypes(include=np.number).columns.tolist()
print(num_columns)
cat_columns = [i for i in data_set if data_set[i].dtypes == 'object']
print(cat_columns)
print(data_set.describe())
print(data_set.info())
print(data_set.isnull().sum()*100/len(data_set))
print(data_set.columns)
print(data_set.head(5))
x = data_set['rm']
y = data_set['medv']
mean_x = x.mean()
mean_y = y.mean()
print(f"The mean of x is {mean_x}")
print(f"The mean of y is {mean_y}")
numerator = ((x-mean_x)*(y-mean_y)).sum()
denominator = ((x-mean_x)**2).sum()
slope = numerator/denominator
print(slope)
intercept = mean_y-(slope*mean_x)
print(intercept)
LinearRegression_for = intercept+(slope*mean_x)
print(f"LinearRegression: {LinearRegression_for}")
# visualization..
# plt.scatter(x=x,y=y, color = 'lightcoral',label = 'Data Points')
# line = intercept+(slope*x)
# plt.plot(x,line,color = 'blue',label = 'Regression Line')
# # plt.legend()
# # for x column change into 2D
# a = data_set[['rm']]
# b = data_set['medv']
# model = LinearRegression()
# model.fit(a,b)
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"slope is {slope} and intercepts is {intercept}")
# rooms = 2
# predict_outcomes = model.predict([[rooms]])
# print(predict_outcomes[0])
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
# print(x_train,x_test,y_train,y_test)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# r_square = r2_score(y_test,y_pred)
# mean_squared_error_1 = mean_squared_error(y_test,y_pred)
# print(r_square,mean_squared_error_1)
# x = data_set[['crim']]
# y = data_set['medv']
# # y = mx +c
# model = LinearRegression()
# model.fit(x,y)
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"slope: {slope},intercept: {intercept}")
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# model.fit(x_train,y_train)
# y_predict = model.predict(x_test)
# r2_score_predict = r2_score(y_test,y_predict)
# mean_squared_error_predict = mean_squared_error(y_test,y_predict)
# print(r2_score_predict,mean_squared_error_predict)
# New ED Data Set
data_set = pd.read_csv("C:\\Users\\vishalappu\\Desktop\\Office\\OEC_PROJECT\\Merge_Data\\OEC_Over_all_06_02.csv")
print(data_set.columns)
num_colum = data_set.select_dtypes(include='number').columns.tolist()
print(num_colum)
cat_colum = data_set.select_dtypes(include=['category']).columns.tolist()
print(cat_colum)
print(data_set.head(10))
print(data_set.info())
# enoding change university categorical into numerical..
from sklearn.preprocessing import LabelEncoder
unique_univerity = data_set['University Name'].unique()
# print(unique_univerity)
le = LabelEncoder()
data_set['University_Encode'] = le.fit_transform(data_set['University Name'])
print(data_set)
print('category:',le.classes_)
print(data_set.info())
B_le = LabelEncoder()
data_set['Branch_Encode'] = B_le.fit_transform(data_set['Branch'])
print(data_set)
num_colum = data_set.select_dtypes(include='number').columns.tolist()
print(num_colum)
print(data_set['Branch_Encode'].unique)
mapping_branch = dict(zip(B_le.classes_,B_le.transform(B_le.classes_)))
# print(mapping_branch)
group_by = data_set[['Branch','Branch_Encode']].groupby('Branch_Encode')
print(group_by.count())
# Linear Regression Part
model = LinearRegression()
x_1 = data_set[['University_Encode']]
y_1 = data_set['Branch_Encode']
model.fit(x_1,y_1)
slope = model.coef_[0]
intercept = model.intercept_
print(f"slope: {slope} and intercepts: {intercept}")
x_train,x_test,y_train,y_test = train_test_split(x_1,y_1,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_prediction = model.predict(x_test)
r2_score_pred = r2_score(y_test,y_prediction)
mean_squared_error_pred = mean_squared_error(y_test,y_prediction)
print(f"r2_pred: {r2_score_pred},mean_pred: {mean_squared_error_pred}")
# visualization..
plt.scatter(x=x_1,y=y_1, color = 'lightcoral',label = 'Data Points')
line = intercept+(slope*x)
plt.plot(x,line,color = 'blue',label = 'Regression Line')
plt.legend()
plt.show()