#importing necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#sample data (house sizes in square feet vs house prices in Rs)
X = np.array([[1200], [1500], [1800], [2000], [2200], [2500]])
Y = np.array([3000000, 3500000, 4000000, 4500000, 5000000, 6000000])

#splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#creating linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

#make predictions
Y_pred = model.predict(X_test)

#evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("Predicted_values:", Y_pred)


