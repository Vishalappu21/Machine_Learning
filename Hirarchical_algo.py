import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification,make_blobs
from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import scipy.cluster.hierarchy as shc
'''Import data'''
def data(file_name):
    try:
        data_xl = pd.read_excel(file_name)
        data_frame = pd.DataFrame(data_xl)
        return data_frame
    except FileNotFoundError:
        print(f'The file {file_name} is not Found..')
        return None
df = data('NM_OEC_CPR_23rd Feb.xlsx')
# print(df.head(5))

'''Data Exploration'''
def data_explore(x):
    df_info = x.info()
    df_describe = x.describe()
    return{
        'data_info':df_info,
        'data_describe':df_describe
    }
df_expo = data_explore(df)
# print(df_expo)

'''Identify Missing Values'''
def check_data(y):
    null_info = y.isnull().sum()
    return null_info
df_chec_values = check_data(df)
# print(df_chec_values)

'''Num and Cat column'''
def num_cat_colum(a):
    # data = a['add_data']
    columns_header = a.columns
    num_columns = a.select_dtypes(include = 'number')
    cat_column = a.select_dtypes(include = 'object')
    return{
        'col_header':columns_header,
        'numerical_column':num_columns,
        'categorical_column':cat_column
    }
variuos_column = num_cat_colum(df)
# print('\nThe column in data is:',variuos_column['col_header'])
# print('\nThe numerical column in data is:',variuos_column['numerical_column'])
# print('\nThe Categorical column in data is:',variuos_column['categorical_column'])

'''Add values'''
def add_values(df,num_colm,cat_colm):
    # cat_colum = z['categorical_column']
    for j in num_colm:
        df[j] = df[j].fillna(df[j].mean())
    for i in cat_colm:
        df[i] = df[i].fillna(df[i].mode()[0])
    return df
new_df = add_values(df=df,num_colm=variuos_column['numerical_column'],cat_colm=variuos_column['categorical_column'])
# print(new_df)
# print(new_df['certification_1_activities'].value_counts())

'''change cat_to_num'''
def Kmeans(data_input):
    mapping = {'Yes':1,'No':0}
    num_data = data_input['percentage_of_course_completion']
    cat_colum = data_input['certification_1_activities'].map(mapping)
    feature = pd.DataFrame({
        'percentage_of_course_completion':num_data,
        'certification_1_activities':cat_colum
    })
    feature = feature.dropna()

    k = 3
    k_m = KMeans(n_clusters=k,random_state=23)
    labels = k_m.fit_predict(feature)
    data_input.loc[feature.index,'cluster']=labels
    plt.scatter(feature['percentage_of_course_completion'],
                feature['certification_1_activities'],
                c=labels, cmap='viridis', s=50)
    plt.scatter(k_m.cluster_centers_[:,0], k_m.cluster_centers_[:,1],
                c='red', marker="*", s=200)
    plt.xlabel("Course Completion %")
    plt.ylabel("Certification Activity (0/1)")
    plt.title("KMeans Clustering: Completion vs Certification")
    # plt.show()
    cols_to_show = ['Student Name','percentage_of_course_completion','certification_1_activities']
    if 'Cluster' in data_input.columns:
        cols_to_show.append('Cluster')


    return {
        'Labels': labels,
        'Centers': k_m.cluster_centers_,
        'Clustered_Data': data_input[cols_to_show].head(10)
    }
k_means_output = Kmeans(new_df)
print("Cluster centers:\n", k_means_output["Centers"])
print("Labels for each student:\n", k_means_output["Labels"])
print(k_means_output["Clustered_Data"].head())

def unit_wise(data_input):
    unit_1 = data_input['unit_1_35_activities']
    unit_2 = data_input['unit_2_35_activities']
    unit_3 = data_input['unit_3_35_activities']
    unit_4 = data_input['unit_4_35_activities']
    unit_5 = data_input['unit_5_36_activities']
    data_feature = {
        'unit_1_35_activities':unit_1,
        'unit_2_35_activities':unit_2,
        'unit_3_35_activities':unit_3,
        'unit_4_35_activities':unit_4,
        'unit_5_36_activities':unit_5
    }
    df = pd.DataFrame(data_feature)
    features = df[['unit_1_35_activities','unit_2_35_activities','unit_3_35_activities','unit_4_35_activities','unit_5_36_activities']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    k = 3
    kmeans = KMeans(n_clusters=k,random_state=23)
    clusters = kmeans.fit_predict(scaled_features)
    df['clusters'] = clusters
    plt.scatter(df['unit_1_35_activities'],df['unit_2_35_activities'],
                c=clusters, cmap='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
                c='red', marker="*", s=200)
    plt.xlabel("Course Completion %")
    plt.ylabel("Certification Activity (0/1)")
    plt.title("KMeans Clustering: Completion vs Certification")
    # plt.show()
    print(df)
    print("\nCluster Centers (scaled):")
    print(kmeans.cluster_centers_)
print(unit_wise(data_input=new_df))

# x,y = make_blobs(n_samples=10,n_features=2,centers=3,random_state=23)
# print(x)
# print(y.shape)
# linked = linkage(x,method='ward')
# print(linked)
# plt.figure(figsize=(10,8))
# dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=True)
# plt.show()
# hc = AgglomerativeClustering(n_clusters=3,linkage='ward')
# label = hc.fit_predict(x)
# plt.scatter(x[:,0],x[:,1], c=label, cmap='viridis', s=50)
# plt.title("Agglomerative Clustering Output")
# plt.show()
# print("Predicted cluster labels:", label)

def hierar(data_input):
    speakin_1 = data_input['speaking_practice_1_activities']
    speakin_2 = data_input['speaking_practice_1_activities_1']
    speakin_3 = data_input['speaking_practice_1_activities_2']
    speakin_4 = data_input['speaking_practice_1_activities_4']
    data_frame = {
        'speaking_practice_1_activities':speakin_1,
        'speaking_practice_1_activities_1':speakin_2,
        'speaking_practice_1_activities_2':speakin_3,
        'speaking_practice_1_activities_4':speakin_4
    }
    df = pd.DataFrame(data_frame)
    sample_df = df.sample(n=500,random_state=42)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sample_df)
    # print(f'\n scaled values of the data: {scaled_data}')
    z = linkage(scaled_data,method='ward',metric='euclidean')
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    shc.dendrogram(Z=z)
    plt.xlabel('Students')
    plt.ylabel('Euclidean distance')
    plt.show()
    hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = hc.fit_predict(scaled_data)
    df['clusters'] = labels
    return df
clustred_df = hierar(new_df)
print(clustred_df.head(5))
