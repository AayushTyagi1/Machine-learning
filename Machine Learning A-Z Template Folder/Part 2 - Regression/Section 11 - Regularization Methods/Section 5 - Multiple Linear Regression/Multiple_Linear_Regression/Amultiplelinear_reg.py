#Aayush Tyagi 2013206
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('startup.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Labelx=LabelEncoder()
X[:,3]=Labelx.fit_transform(X[:,3])
print(X)
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3 , random_state = 0)
print(X)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#avoid dummy variable
X=X[:,1:]

#fitting
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

#pred
y_pred=reg.predict(X_test)
#Backward Elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
print(X)
X_opt = X[:,[0,1,2,3,4,5]]
reg_OLS=sm.OLS(endog=y,exog=X_opt).fit()
reg_OLS.summary()

import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
print(X)
X_opt = X[:,[0,1,2,3,4]]
reg_OLS=sm.OLS(endog=y,exog=X_opt).fit()
reg_OLS.summary()