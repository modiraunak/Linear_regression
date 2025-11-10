from  sklearn.mixture import GaussianMixture
import numpy as np
#Sample Data
X = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],[7, 85], [8, 90],[9,95],[10,100]])
# Perform Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=2, random_state=0)  # n_components is the number of mixture components
gmm.fit(X)
# Get the cluster labels
labels = gmm.predict(X)
print("labels: ", labels)
# Get the probabilities of each point belonging to each cluster
probabilities = gmm.predict_proba(X)
print("Probabilities: ", probabilities) 
