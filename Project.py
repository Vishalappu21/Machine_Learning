'''Add end-to-end Bank Marketing ML pipeline
- Imported and cleaned dataset (bank.csv)
- Performed exploratory analysis with Seaborn visualizations
- Encoded categorical features using OneHotEncoder
- Scaled numerical features with MinMaxScaler and StandardScaler
- Engineered interaction features (balance × housing, balance × loan)
- Implemented Logistic Regression for deposit classification
- Implemented Linear Regression for balance prediction
- Evaluated models with accuracy, confusion matrix, classification report, R², MSE, RMSE
- Visualized regression line overlay on scatter plot'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
'''Import DataSet..'''
data_set = pd.read_csv('bank.csv') 
print(data_set)

'''Data_Cleaning...'''
d_info = data_set.info()
# print(d_info)
d_describe = data_set.describe()
# print(d_describe)
isnull_check = (data_set.isnull().sum())*100/len(data_set)
# print(isnull_check)

'''- Check correlations between numerical features.'''
identify_num_col = data_set.select_dtypes(include=np.number).columns.tolist()
# print(identify_num_col)
# num_col = [i for i in data_set if data_set[i].dtypes != 'object']
# print(num_col)
check_corr = data_set[identify_num_col].corr()
# print(check_corr)
# Polt_Distribution..
plot_dist = data_set[['age','balance','duration']]
# sns.barplot(data=plot_dist)
# plt.show()
# pd.crosstab(data_set['job'],data_set['education']).plot(kind='bar',stacked=True)
sns.set(style='whitegrid')
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
ax1 = sns.countplot(x='job',data=data_set,palette='viridis')
plt.title('Job Distribution')
plt.xticks(rotation=45)
for bar in ax1.patches:
    height = bar.get_height()
    x = bar.get_x() + bar.get_width()/2  
    ax1.text(x, height, str(height), ha='center', va='bottom')
plt.subplot(1, 3, 2)
ax2 = sns.countplot(x='marital', data=data_set, palette='magma')
plt.title('Marital Status Distribution')
for p in ax2.patches:
    ax2.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='bottom')
plt.subplot(1, 3, 3)
ax3 = sns.countplot(x='education', data=data_set, palette='coolwarm')
plt.title('Education Distribution')
for p in ax3.patches:
    ax3.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='bottom')
# plt.tight_layout()
# plt.show()

'''- Feature Engineering'''
# Encode...
# The encoder “finds the object (category)” and marks its place with a 1. Everywhere else is 0.
identify_cat_col = data_set.select_dtypes(include='object').columns.tolist()
print(identify_cat_col)
encode = OneHotEncoder(sparse_output=False)
one_hot_encode = encode.fit_transform(data_set[identify_cat_col])
encode_data_set = pd.DataFrame(one_hot_encode,columns=encode.get_feature_names_out(identify_cat_col))
df_encoder = pd.concat([data_set,encode_data_set],axis=1)
df_encode = df_encoder.drop(identify_cat_col,axis=1)
print(df_encode)

'''- Scale numerical features'''
# MinmaxScaler..
data = data_set[['age','balance','duration']]
scaler = MinMaxScaler()
scaler_fit = scaler.fit_transform(data)
scaler_df = pd.DataFrame(scaler_fit,columns=['age_scaled','balance_scaled','duration_sccaled'])
data_scaled = pd.concat([data_set,scaler_df],axis=1)
print(data_scaled)

#z-score scaler
data = data_set[['age','balance','duration']]
z_scale = StandardScaler()
z_scale_fit = z_scale.fit_transform(data)
z_scale_df = pd.DataFrame(z_scale_fit,columns=['age_z_scale','balance_z_scale','duration_z_scale'])
z_scaled = pd.concat([data_set,z_scale_df],axis=1)
print(z_scaled)

#Polynomial Feactures..
encode = pd.get_dummies(data_set[['housing','loan']],drop_first=True)
data_num = data_set[['age','balance','duration']]
data_encode = pd.concat([data_num,encode],axis=1)
data_encode['balance_housing'] = data_encode['balance'] * data_encode['housing_yes']
data_encode['balance_loan'] = data_encode['balance'] * data_encode['loan_yes']
print(data_encode)
print(data_encode['loan_yes'].value_counts())
print(data_encode['housing_yes'].value_counts())

'''- Logistic Regression (Classification)'''
# - Target: deposit (yes/no).
# formula = 1/(1+math.exp(power_e))
# power_e = -(y-(slope*x))
print(data_set['deposit'].value_counts())
data_set.loc[data_set['deposit']=='yes','Deposit_num']=1
data_set.loc[data_set['deposit']=='no','Deposit_num']=0
print(data_set['Deposit_num'].value_counts())
x = data_set[['age']]
y = data_set['Deposit_num']
model = LogisticRegression()
model.fit(x,y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"slope: {slope} and intercept: {intercept}")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
model.fit(x_train,y_train)
model_predction = model.predict(x_test)
accuracy_score_test = accuracy_score(y_test,model_predction)
confusion_matrix_test = confusion_matrix(y_test,model_predction)
classification_report_test = classification_report(y_test,model_predction)
print(f"accuracy_score_test:\n {accuracy_score_test}")
print(f"classification_report_test:\n {classification_report_test}")
print(f"confusion_matrix_test:\n {confusion_matrix_test}")

'''Linear Regression (Regression)'''
col = data_set.columns
# print(col)
a = data_set[['age']]
b = data_set['balance']
model_fit = LinearRegression()
model_fit.fit(a,b)
slope_1 = model_fit.coef_[0]
intercept_1 = model_fit.intercept_
print(f"slope_for_linear: {slope_1} and intercept_for_linear: {intercept_1}")
x_train,x_test,y_train,y_test = train_test_split(a,b,test_size=0.2,random_state=42)
model_fit.fit(x_train,y_train)
model_predction_linear = model_fit.predict(x_test)
r2_score_test = r2_score(y_test,model_predction_linear)
mean_squared_error_test = mean_squared_error(y_test,model_predction_linear)
rmse = np.sqrt(mean_squared_error(y_test, model_predction_linear))
print(f"r2_square_test:\n {r2_score_test}")
print(f"mean_square_error:\n {mean_squared_error_test}")
print(f"rmse:\n {rmse}")

'''Visualization'''
x_val = a.values.ravel()
y_val = b.values.ravel()
plt.scatter(x_val,y_val, color = 'lightcoral',label = 'Data Points')
line = intercept_1+(slope_1*x_val)
plt.plot(x_val,line,color = 'blue',label = 'Regression Line')
plt.xlabel("Age")
plt.ylabel("Balance")
plt.title("Linear Regression: Age vs Balance")
plt.legend()
plt.show()