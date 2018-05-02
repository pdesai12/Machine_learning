# Multiple Linear Regression

# Importing the libraries
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import classification_report
import sklearn.metrics
import pydotplus
from scipy import misc

# Importing the dataset


#Get Data from CSV file into dataframe
dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/sgemm_product_dataset/Regressor_train.csv')
X_train = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
y_train = dataset.iloc[:, 17:].values


dataset1 = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/sgemm_product_dataset/Regressor_test.csv')
X_test = dataset1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
y_test = dataset1.iloc[:, 17:].values


"""
dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/AirQualityUCI/AirQualityUCI.csv')
X_train = dataset.iloc[0:9233, [2,3,5,6,7,8,9,10,11,12,13,12]].values
X_test = dataset.iloc[9234:9334, [2,3,5,6,7,8,9,10,11,12,13,12]].values
y_train = dataset.iloc[0:9233, [13]].values
y_test = dataset.iloc[9234:9334,[13]].values
#print(y)

dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/AirQualityUCI/AirQualityUCI.xlsx')
X_test = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
y_test = dataset.iloc[:, 14].values
#print(X)
#print(y)
"""                 



"""
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

#y = np.array(df['Lable'])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:,:])
X_train[:,:] = imputer.transform(X_train[:,:])
imputer = imputer.fit(X_test[:,:])
X_test[:,:] = imputer.transform(X_test[:,:])
imputer = imputer.fit(y_train)
y_train = imputer.transform(y_train)


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
"""

 #Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
absolute_error = mean_absolute_error(y_test,y_pred)
squared_error = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print('r-2 Measure is')
print(r2)
print('Absolute error value is')
print(absolute_error)
print('Sqaured Error is')
print(squared_error)
print('Test Data')
print(y_test[0],y_test[1])
print('Predicted Data')
print(y_pred[0],y_pred[1])
#print(y_test[:])
#print(y_pred[:])
"""
from sklearn.svm import SVR
regressor = SVR(kernel = 'poly')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
#print(y_test[:])
#print(y_pred[:])
"""
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
size = len(y_test)
size = size*2
test = list(range(1,size,2))
pred = list(range(0,size,2))
#print(len(y_test))
#print(len(y_pred))
#print(len(test))
#print(len(pred))
 
plt.bar(test,y_test[:,0],alpha=0.3)
plt.xlabel('Testing Data')
plt.ylabel('Original/Predicted values')
plt.title('Multi_linear')
plt.legend()
 
plt.bar(pred,y_pred[:,0],alpha=0.3)
plt.xlabel('Testing Data')
plt.ylabel('Original/Predicted values')
plt.title('Multi_linear')
plt.legend()
plt.show()

