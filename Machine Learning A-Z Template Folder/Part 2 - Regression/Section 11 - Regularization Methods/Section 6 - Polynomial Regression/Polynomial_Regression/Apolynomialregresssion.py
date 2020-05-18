import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
X=data.iloc[:,1:2].astype(float)
y=data.iloc[:,2].astype(float)

from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg1.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

reg2=LinearRegression()
reg2.fit(X_poly,y)

#Visualise linear regression
plt.scatter(X,y,color='r')
plt.plot(X,reg1.predict(X),color='blue')
plt.title("Linear regression")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

#visualisepolynomial
X_grid=np.arange(min(X['Level']), max(X['Level']), step=0.1,dtype=float)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='r')
plt.plot(X,reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Polynomial regression")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()
reg1.predict(np.array([6.5]).reshape(1,-1))
reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1,-1)))