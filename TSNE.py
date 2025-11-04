from sklearn.manifold import TSNE
import numpy as np
# Sample data (HOURS STUDIED, SCORE)
X = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],[7, 85], [8, 90],[9,95],[10,100]])
# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2,perplexity=5, random_state=42)  # n_components is the number of dimensions to reduce to
X_tsne = tsne.fit_transform(X)
print("Original Data:\n", X)
print("t-SNE Transformed Data:\n", X_tsne)
# Note: t-SNE is primarily used for visualization in 2 or 3 dimensions, so n_components is typically set to 2 or 3.
