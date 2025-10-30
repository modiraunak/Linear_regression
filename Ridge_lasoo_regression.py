# to prevent overfitting in linear regression models we use Ridge and Lasso regression techniques.it prevents overfiting by penalizing large coefficients in the model.
#importing necessary libraries
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#importing necessary libraries
import numpy as np

#sample data (house sizes in square feet vs house prices in Rs)
X = np.array([[1200], [1500], [1800], [2000], [2200], [2500]])
Y = np.array([3000000, 3500000, 4000000, 4500000, 5000000, 6000000])

#splitting data into training and testing sets
x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Ridge Regression
Ridge_model = Ridge(alpha=1.0)
Ridge_model.fit(x_train, Y_train)
Ridge_pred = Ridge_model.predict(x_test)
Ridge_mse = mean_squared_error(Y_test, Ridge_pred)
print("Ridge Regression Mean Squared Error:", Ridge_mse)

#Lasso Regression
Lasso_model = Lasso(alpha=0.1)
Lasso_model.fit(x_train, Y_train)
lasso_pred = Lasso_model.predict(x_test)
lasso_mse = mean_squared_error(Y_test, lasso_pred)
print("Lasso Regression Mean Squared Error:", lasso_mse)
print("Ridge Predicted values:", Ridge_pred)