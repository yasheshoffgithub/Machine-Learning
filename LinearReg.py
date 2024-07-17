import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes= datasets.load_diabetes()
#(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
diabetes_X=diabetes.data[:,np.newaxis,2]
#this is done to see the linear graph , if we use multiple features , MSE will get reduced but we won't be able to plot it
diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-30:]


diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-30:]
model=linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_predict=model.predict(diabetes_X_test)
print("Mean squared error is: ",mean_squared_error(diabetes_y_test,diabetes_y_predict))
print("Weights: ",model.coef_)
print("Intercept: ",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_y_test)
plt.plot(diabetes_X_test,diabetes_y_predict)
plt.show()