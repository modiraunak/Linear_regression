#import neccessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
#sample data (no of hours studied vs pass or fail)
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = fail, 1 = pass
#splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#initializing and training the model
model = LogisticRegression()
model.fit(x_train, y_train)
#making predictions
y_pred = model.predict(x_test)
#evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
#displaying predictions
for i, pred in enumerate(y_pred):
    result = "Pass" if pred == 1 else "Fail"
    print(f"Predicted result for studying {x_test[i][0]} hours: {result}") 