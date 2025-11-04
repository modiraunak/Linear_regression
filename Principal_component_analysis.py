#importing necessary libraries
from sklearn.decomposition import PCA
import numpy as np
# Sample data (HOURS STUDIED, SCORE)
X = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],[7, 85], [8, 90],[9,95],[10,100]])
# Perform Principal Component Analysis (PCA)
pca = PCA(n_components=2)  # n_components is the number of principal components 
X_pca = pca.fit_transform(X)
print("Original Data:\n", X)
print("PCA Transformed Data:\n", X_pca)
# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
# Components
components = pca.components_
print("PCA Components:\n", components)
# Reconstructing the original data from the PCA transformed data
X_reconstructed = pca.inverse_transform(X_pca)