# K-Means Clustering

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the restaurant_dataset
restaurant_dataset = pd.read_csv('Restaurant_Customers.csv')
X = restaurant_dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 39)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method: ')
plt.xlabel('cluster number: ')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the restaurant_dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 39)
kmeans_y = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[kmeans_y == 0, 0], X[kmeans_y == 0, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[kmeans_y == 1, 0], X[kmeans_y == 1, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.scatter(X[kmeans_y == 2, 0], X[kmeans_y == 2, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plt.scatter(X[kmeans_y == 3, 0], X[kmeans_y == 3, 1], s = 100, c = 'green', label = 'Cluster 4')
plt.scatter(X[kmeans_y == 4, 0], X[kmeans_y == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('cluster number: ')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()