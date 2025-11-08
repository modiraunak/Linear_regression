from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# Sample data
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_labels, X_unlabeled, y_labels, _ = train_test_split(X, y, test_size=0.5, random_state=42)
# Introduce some unlabeled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_labels, y_labels)
# Self-training loop
for _ in range(5):  # 5 iterations of self-training
    # Predict Probabilities on unlabeled data
    probs = model.predict_proba(X_unlabeled)
    # Select high-confidence predictions
    high_confidence_indices = np.where(np.max(probs, axis=1) > 0.9)[0]
    #add high-confidence predictions to labeled dataset
    x_labbeled = np.vstack((X_labels, X_unlabeled[high_confidence_indices]))
    y_labbeled = np.hstack((y_labels, np.argmax(probs[high_confidence_indices], axis=1)))
    # Remove newly labeled data from unlabeled dataset
    X_unlabeled = np.delete(X_unlabeled, high_confidence_indices, axis=0)
    # Retrain the model
    model.fit(x_labbeled, y_labbeled)
    X_labels, y_labels = x_labbeled, y_labbeled
# Evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X_labels, y_labels, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final accuracy after self-training: {accuracy:.2f}")
    