from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
#Sample data (HOURS STUDIED, SCORE)
x = np.array([[1, 50], [2, 60], [3, 70], [6, 80], [7, 90], [8, 100]])
y = np.array([0, 0, 0, 1, 1, 1])  # 0: Fail, 1: Pass
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# inalize and train the KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
# Make predictions
y_pred = model.predict(x_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
