from sklearn.ensemble import IsolationForest
import numpy as np 

#Sample Data 
x = 0.3 * np.random.randn(100, 2)
x_train = np.r_[x + 2, x - 2] #Create a dataset with points 

#test data including some outliears 
x_test = np.r_[x + 2,x - 2, np.random.uniform(low = 6, high = 6, size=(20,2))]

#initialize and train the model 
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(x_train)

#predict on data 
predictions = model.predict(x_test)

#display the results 
print("Predictions",predictions)




