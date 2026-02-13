import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_set = pd.read_csv("diabetes.csv")
print(data_set)
cat_col = [i for i in data_set if data_set[i].dtype == 'object']
num_col = [i for i in data_set if data_set[i].dtype != 'object']
print(f'Cat_Col:{cat_col}')
print(f'Num_Col:{num_col}')
duplicate_check = data_set.duplicated()
print(duplicate_check)
check_missing_values = data_set.isnull().sum()
print(check_missing_values)
row_duplicate = data_set['Glucose'].isnull()
print(f"Gulcose:{row_duplicate}")
# unique_check = data_set[num_col].nunique()
# print(unique_check)
#Mean:
mean_check = data_set['Glucose'].mean()
median_check = data_set['Glucose'].median()
std_check = data_set['Glucose'].std()
print(median_check)
print(mean_check)
print(std_check)
#correlation 
corr_values = data_set['Age'].corr(data_set['BMI'])
print(corr_values)
plt.hist(x=data_set['Insulin'],bins='auto')
# plt.show()
# - Write code to normalize the Glucose column using Min-Max scaling.
# Glucose_data = data_set['Glucose']
# min_glu = min(Glucose_data)
# max_glu = max(Glucose_data)
# normalize = max_glu - min_glu
# print(normalize)
min_val = data_set['Glucose'].min()
max_val = data_set['Glucose'].max()
data_set['Glucose_Normalized'] = (data_set['Glucose'] - min_val) / (max_val - min_val)
print(data_set[['Glucose','Glucose_Normalized']].head())
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
scaler = MinMaxScaler()
glucose_min_max = scaler.fit_transform(data_set[['Glucose']])
data_set['Glucose Normalize'] = glucose_min_max
print(data_set[['Glucose','Glucose Normalize']])
s_scaler = StandardScaler()
glucose_scaler = s_scaler.fit_transform(data_set[['Glucose']])
data_set['Glucose standardscaler'] = glucose_scaler
print(data_set[['Glucose','Glucose standardscaler']])
print(data_set.columns)
data_set.loc[data_set['BMI']<18.5,'BMI_Category']='Under_weight'
data_set.loc[(data_set['BMI']>=18.5)&(data_set['BMI']<=24.5),'BMI_Category']='Healthy_Wight'
data_set.loc[(data_set['BMI']>=25.0)&(data_set['BMI']<=29.9),'BMI_Category']='Over_Wight'
data_set.loc[(data_set['BMI']>=30.0)&(data_set['BMI']<=39.9),'BMI_Category']='Obesity'
data_set.loc[data_set['BMI']>=40,'BMI_Category']='Severe_Weight'
print(data_set.head(5))
print(data_set[['BMI','BMI_Category']])
cat_col = [i for i in data_set if data_set[i].dtype == 'object']
num_col = [i for i in data_set if data_set[i].dtype != 'object']
print(f"cat_col: \n{cat_col}")
print(f"Num_col: \n{num_col}")
from sklearn.preprocessing import OneHotEncoder
#identify categorical column
cate_colum = data_set.select_dtypes(include=['object']).columns.tolist()
# print(cate_colum)
encoder = OneHotEncoder(sparse_output=False)
one_hot_enocoder = encoder.fit_transform(data_set[cate_colum])
one_hot_df = pd.DataFrame(one_hot_enocoder,columns=encoder.get_feature_names_out(cate_colum))
df_encoder = pd.concat([data_set,one_hot_df],axis=1)
df_encode = df_encoder.drop(cate_colum,axis=1)
print(df_encode)
#unique_values:..
unique = data_set['BMI_Category'].nunique()
unique_name = data_set['BMI_Category'].unique()
print(unique)
print(unique_name)
print(data_set.describe())
print(data_set.columns)
missing_values = data_set.isnull().sum()*100 /len(data_set)
print(missing_values)

#Visualizations
x = data_set['Glucose']
y = data_set['Age']
# plt.scatter(x=y,y=x)
# plt.show()
import seaborn as sns
# grouped_by = data_set.groupby('BMI')['Outcome'].sum()
# plt.figure(figsize=(8,6))
# grouped_by.plot(kind='box',color='red')
# plt.xlabel('BMI')
# plt.ylabel('Total Outcomes')
# plt.xticks(rotation=0)
# plt.show()
print(data_set['Outcome'].unique())
# plt.bar(data_set['Outcome'].value_counts(),color = 'violet')
# data_set['Outcome'].value_counts(sort=False).plot.bar()
# plt.title('Outcome')
# plt.show()
num_col = [i for i in data_set if data_set[i].dtype != 'object']
print(f"Num_col: \n{num_col}")
# num_columns = data_set.select_dtypes(include=np.number).columns.tolist()
# print(num_columns)
import plotly.express as px # type: ignore
num_columns = data_set.select_dtypes(include=np.number).columns.tolist()
print(num_columns)
corr_max = data_set[num_columns].corr()
fig = px.imshow(corr_max,text_auto=True,color_continuous_scale='Viridis')
fig.show()
aver = data_set['Glucose'].mean()
print(aver)
count_oc = data_set['Outcome'].value_counts()
print(count_oc)
aver_oc = data_set[['Glucose','Outcome']].groupby('Outcome').mean().plot.bar()
print(aver_oc)
plt.show()
x = data_set.groupby('Glucose')['Outcome'].mean()
print(x.head(10))