#import neccessary libraries
from sklearn.cluster import DBSCAN
import numpy as np

# Sample data (HOURS STUDIED, SCORE)
X = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],[7, 85], [8, 90],[9,95],[10,100]])
# Perform DBSCAN clustering
dbscan = DBSCAN(eps=5, min_samples=2)  # eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other
dbscan.fit(X)

# Get the cluster labels
labels = dbscan.labels_
print("labels: ",labels)
