#Aayush Tyagi 2013206
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values


##fitting
from sklearn.ensemble import RandomForestRegressor as RF
regressor=RF(n_estimators=10,random_state=0)
regressor.fit(X,y)

#predicting
y_pred = regressor.predict(np.array([[6.5]]))

plt.scatter(X,y,color='r')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Random forest')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

X_grid=np.arange(min(X), max(X), step=0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='r')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random Forest Model')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()