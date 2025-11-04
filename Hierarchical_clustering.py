from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np 

# Sample data (HOURS STUDIED, SCORE)
X = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],[7, 85], [8, 90],[9,95],[10,100]])
# Perform hierarchical clustering using the 'ward' method
Z = linkage(X, method='ward') # ward minimizes the variance within clusters
# Create a dendrogram to visualize the hierarchical clustering
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=[f'Stud {i+1}' for i in range(X.shape[0])])
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Students')
plt.ylabel('Distance')
plt.show()
