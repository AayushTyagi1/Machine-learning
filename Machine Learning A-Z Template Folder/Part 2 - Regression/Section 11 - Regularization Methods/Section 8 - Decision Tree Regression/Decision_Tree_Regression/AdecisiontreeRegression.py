import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].astype(float)
y=data.iloc[:,2:3].astype(float)

from sklearn.tree import DecisionTreeRegressor as DTS
regressor=DTS(random_state=0)
regressor.fit(X,y)

y_pred = regressor.predict(np.array([[6.5]]))

plt.scatter(X,y,color='r')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Decision Tree')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

X_grid=np.arange(min(X['Level']), max(X['Level']), step=0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='r')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('SVR Model')
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()