import numpy as np
#generate syntehtic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (intercept) to feature matrix
X_b = np.c_[np.ones((100, 1)), X]

# Mathematical formula for lineaqr Regression
def predict(x,theta):
    return np.dot(x, theta)
#inalize model parameters
theta = np.random.randn(2, 1)
alpha = 0.1
iterations = 1000
# Use Gradient Descent to optimize the model parameters
def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = predict(x, theta)
        errors = predictions - y
        gradient = (1/m) * np.dot(x.T, errors)
        theta -= alpha * gradient
    return theta 
# Calculation of Evaluation Metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

# Perform gradient descent to find optimal theta
theta_optimized = gradient_descent(X_b, y, theta, alpha, iterations)

# Predictions and evaulations
y_pred = predict(X_b, theta_optimized)
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred)

print("Optimized parameters (theta):",theta_optimized)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

