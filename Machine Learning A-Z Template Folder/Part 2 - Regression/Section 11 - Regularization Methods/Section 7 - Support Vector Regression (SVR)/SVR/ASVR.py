import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].astype(float)
y=data.iloc[:,2:3].astype(float)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor =SVR(kernel= 'rbf')
regressor.fit(X,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X,y,color='r')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('SVR')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

X_grid=np.arange(min(X), max(X), step=0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='r')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('SVR Model')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()