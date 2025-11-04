#import neccessary libraries
from sklearn.cluster import KMeans
import numpy as np 

# Sample data (HOURS STUDIED, SCORE)
X = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],[7, 85], [8, 90],[9,95],[10,100]])
# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Get the cluster labels
centeroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(f"Clusters Centers:\n{centeroids}")
print(f"Labels:\n{labels}")
# Predict the cluster for new data points
new_data = np.array([[3, 60], [8, 85]])
predictions = kmeans.predict(new_data)