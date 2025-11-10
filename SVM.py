from sklearn.svm import OneClassSVM
import numpy as np 

x = 0.3 * np.random.randn(100,2)
x_train = np.r_[x + 2,x - 2]

#new  test data with outliners
x_test = np.r_[x + 2, x - 2, np.random.uniform(low=6, high=6, size =(20,2))]

# initalize the and train the model 
model = OneClassSVM(gamma = 'auto', nu = 0.1)
model.fit(x_train)

#predict the model on test data 
predictions = model.predict(x_test)

#display Results 
print("Predictions", predictions)