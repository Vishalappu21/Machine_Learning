import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree,DecisionTreeRegressor,export_text,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestClassifier

'''Import Data'''
bank_data = pd.read_csv('bank.csv')
print(bank_data.head())

'''Using encode or Dummy convert cat_column into Num_colu which gives Error'''
# print(bank_data['job'].value_counts())
category_colum = bank_data.select_dtypes(include='object').columns.tolist()
# print(category_colum)
cat_colum = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# dummy = pd.get_dummies(bank_data['job'],drop_first=True)
# print(dummy)
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoder = encoder.fit_transform(bank_data[cat_colum])
one_hot_df = pd.DataFrame(one_hot_encoder,columns=encoder.get_feature_names_out(cat_colum))
df_encoder = pd.concat([bank_data,one_hot_df],axis=1)
# print(df_encoder.columns)
df_encode = df_encoder.drop(cat_colum,axis=1)
# df_encode['deposit'] = LabelEncoder().fit_transform(df_encode['deposit'])
df_encode['deposit'] = bank_data['deposit'].map({'yes':1, 'no':0})
print(df_encode.columns)
print(bank_data.columns)
print(df_encode.head())


'''Decision Tree....'''
x = df_encode.drop('deposit',axis=1)
# print(x.columns)
y = df_encode['deposit']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
clf = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=42)
clf.fit(x_train,y_train)
reg = DecisionTreeRegressor(criterion='squared_error',max_depth=3,random_state=42)
reg.fit(x_train,y_train)
tree_rule = export_text(clf,feature_names=list(x.columns))
print(tree_rule)

'''Visualize'''
# plt.figure(figsize=(20,10))
# plot_tree(reg,feature_names=x.columns,class_names=['No_Dep','Depo'],filled=True,rounded=True)
# plt.show()
# dot_data = export_graphviz(clf, 
#                            out_file=None, 
#                            feature_names=x.columns,
#                            class_names=['No Deposit','Deposit'],
#                            filled=True, 
#                            rounded=True,
#                            special_characters=True)

# graph = graphviz.Source(dot_data)
# graph.render("decision_tree")  
# graph.view()

'''RandomForestClassifier'''
rf = RandomForestClassifier(n_estimators=100,max_depth=None,max_features='sqrt',random_state=42)
rf.fit(x_train,y_train)
# Predctions
y_predction = rf.predict(x_test)
print("Train Accuracy:", rf.score(x_train, y_train))
print("Test Accuracy:", accuracy_score(y_test, y_predction))
importances = pd.Series(rf.feature_importances_, index=x.columns)
print(importances.sort_values().head(10))