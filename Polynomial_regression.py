#importing necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#sample data(experience vs salary)
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([30000, 35000, 50000, 60000, 80000])

#splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#transforming features to polynomial features
poly_features = PolynomialFeatures(degree=2) 
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)
#training the model
model = LinearRegression()
model.fit(x_train_poly, y_train)
#making predictions
y_pred = model.predict(x_test_poly)
#evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
#displaying predictions
for i, pred in enumerate(y_pred):
    print(f"Predicted salary for experience {x_test[i][0]} years: {pred}")
    
