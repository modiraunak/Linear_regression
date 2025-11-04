#importing necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Sample data (HOURS STUDIED, SCORE)
X = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 75], [6, 80],[7, 85], [8, 90],[9,95],[10,100]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0: Fail, 1: Pass

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the Native Bytes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")