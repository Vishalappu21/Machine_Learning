import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from pandas.core.common import random_state

'''Import data'''
data_df = pd.read_csv('bank.csv')
print(data_df.head(5))

'''Change cat_col into Num_col'''
category_colum = data_df.select_dtypes(include='object').columns.tolist()
# print(category_colum)
cat_colum = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# dummy = pd.get_dummies(data_df['job'],drop_first=True)
# print(dummy)
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoder = encoder.fit_transform(data_df[cat_colum])
one_hot_df = pd.DataFrame(one_hot_encoder,columns=encoder.get_feature_names_out(cat_colum))
df_encoder = pd.concat([data_df,one_hot_df],axis=1)
# print(df_encoder.columns)
df_encode = df_encoder.drop(cat_colum,axis=1)
# df_encode['deposit'] = LabelEncoder().fit_transform(df_encode['deposit'])
df_encode['deposit'] = data_df['deposit'].map({'yes':1, 'no':0})
print(df_encode.columns)
# print(data_df.columns)
print(df_encode.head())

'''boosting algorithm'''
# model = XGBClassifier()
x = df_encode.drop('deposit',axis=1)
y = df_encode['deposit']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
xgf = XGBClassifier(
    n_estimators=100,
    learning_rate = 0.1,
    max_depth = 3,
    random_state=42)
xgf.fit(x_train,y_train)
y_pred = xgf.predict(x_test)
print("Train Accuracy:", xgf.score(x_train, y_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

importances = pd.Series(xgf.feature_importances_, index=x.columns)
print(importances.sort_values(ascending=False).head(10))
