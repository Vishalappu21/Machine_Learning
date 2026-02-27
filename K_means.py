import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

'''Structured Data'''
data = {
    'Customer': ['A','B','C','D','E','F','G','H','I','J'],
    'Annual_Income': [15, 16, 17, 35, 40, 42, 60, 65, 70, 85],
    'Spending_Score': [39, 81, 6, 77, 40, 50, 60, 20, 90, 30]
}
def change_data():
    data_df = pd.DataFrame(data)
    return {
        'New_data':data_df
    }
changed_data = change_data()
print(changed_data)

def num_column(a):
    column = a['New_data']
    numeric_column = column.select_dtypes(include = 'number')
    return {
        'numerical_column':numeric_column
    }
num_output = num_column(changed_data)
print(num_output)

def k_means(b):
    data_clustring = b['numerical_column']
    k = 3
    k_m = KMeans(n_clusters=k,random_state=23)
    k_m.fit(data_clustring)
    label = k_m.labels_
    plt.scatter(data_clustring.values[:, 0], data_clustring.values[:, 1], c=label, cmap='viridis', s=30)
    plt.scatter(k_m.cluster_centers_[:,0],k_m.cluster_centers_[:,1],c='red',marker="*",s=200)
    plt.title('KMeans Output')
    plt.show()
    return {
        "labels": label, "centers": k_m.cluster_centers_
    }
k_means_output = k_means(num_output)
print("Cluster centers:\n", k_means_output["centers"])
print("Labels for each customer:\n", k_means_output["labels"])

'''Unstructured Data'''
x,y = make_blobs(n_samples=200,n_features=2,centers=5,random_state=23)
print(x.shape)
print(y)
plt.scatter(x[:,0],x[:,1],c=y,cmap='viridis_r',s=30)
plt.grid(True)
# plt.show()
k = 5
k_m = KMeans(n_clusters=k,random_state=23)
k_m.fit(x)
label = k_m.labels_
print(label)
plt.scatter(x[:,0],x[:,1],c=label,cmap='viridis',s=30)
plt.scatter(k_m.cluster_centers_[:,0],k_m.cluster_centers_[:,1],c='red',marker="*",s=200)
plt.title('KMeans Output')
plt.show()


# x,y = make_blobs(n_samples=10,n_features=2,centers=3,cluster_std=0.60,random_state=23)
# print('\n the values of x')
# print(x)
# print('\n the values of y')
# print(y)
# print('the shape of x is:',x.shape)
# print('the shape of y is',y.shape)
# print(x[0])
# print(y[0])
# for i in range(10):
#     print(f"point {i}: {x[i]} label: {y[i]}")
# a = x[:,0]
# print(a)
# b = x[:,1]
# print(b)
# plt.scatter(a,b,c=y,cmap='viridis',s=30)
# plt.title("Blob's data visula")
# plt.show()
# c = KMeans(n_clusters=3,random_state=23)
# c.fit(x)
# labels = c.labels_
# plt.scatter(a,b,c=labels,cmap='viridis',s=30)
# plt.scatter(c.cluster_centers_[:,0],c.cluster_centers_[:,1],c='red',marker="*",s=200)
# plt.title('KMeans Output')
# plt.show()
# k = 2
# np.random.seed(23)


# fig = plt.figure(0)
# plt.grid(True)
# plt.scatter(x[:,0],x[:,1])
# plt.show()
# k = 2
# clusters = {}
# np.random.seed(23)
# for i in range(k):
#     center = 2*(2*np.random.random((x.shape[1],))-1)
#     points = []
#     cluster={
#         'center':center,
#         'points':[]
#     }
#     clusters[i]=cluster
# print(clusters)
# plt.scatter(x[:,0],x[:,1])
# plt.grid(True)
# for j in clusters:
#     center = clusters[j]['center']
#     plt.scatter(center[0],center[1],marker='*',c='red')
# plt.show()

# data = {
#     'std_name':['A','B','C','D','E','F'],
#     'Maths_score':[90,88,30,28,60,58],
#     'Science_score':[85,80,25,20,65,62]
# }
# data_df = pd.DataFrame(data)
# print(data_df)
# print(type(data_df))
# numeric_column = data_df.select_dtypes(include='number')
# print(numeric_column)
# Kmeans = KMeans(n_clusters=2,random_state=42)
# Kmeans.fit(numeric_column)
# y_kmeans = Kmeans.predict(numeric_column)
# new_df = data_df.assign(cluster=y_kmeans)
# print(new_df)
# centers = Kmeans.cluster_centers_
# print(centers)
# plt.scatter(numeric_column.values[:, 0], numeric_column.values[:, 1], c=y_kmeans, s=50, cmap='viridis')
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='x')
# plt.title("K-Means Clustering Result")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()