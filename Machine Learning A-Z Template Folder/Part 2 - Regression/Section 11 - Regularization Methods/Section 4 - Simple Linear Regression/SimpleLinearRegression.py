#Aayush Tyagi 2013206
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3 , random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

##fitting
from sklearn.linear_model import LinearRegression
rg=LinearRegression()
rg.fit(X_train,y_train)

#predicting
y_pred = rg.predict(X_test)
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,rg.predict(X_train),color='blue')
plt.title('Salary VS Experience(Training set)')
plt.xlabel('Year of ecperience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,rg.predict(X_train),color='blue')
plt.title('Salary VS Experience(Test set)')
plt.xlabel('Year of ecperience')
plt.ylabel('Salary')
plt.show()