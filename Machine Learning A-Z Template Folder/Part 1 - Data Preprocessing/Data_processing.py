#Aayush Tyagi 2013206

#importing the libraries
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

#importing the dataset
dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:, :-1].values
print(X)
Y=dataset.iloc[:,3].values
print(Y)

#Taking care of missing values
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit(X[:,1:3])
X[:,1:3]=imp.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Labelx=LabelEncoder()
X[:,0]=Labelx.fit_transform(X[:,0])
print(X)
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'                        
)

X = np.array(ct.fit_transform(X), dtype=np.float)
Labely=LabelEncoder()
Y=Labely.fit_transform(Y)
print(Y)

#Splitting the data into training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler as SS
sc_X=SS()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.transform(x_test)