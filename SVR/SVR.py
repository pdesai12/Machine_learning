import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.decomposition import PCA
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


#Get Data from CSV file into dataframe
dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/sgemm_product_dataset/Regressor_train.csv')
X_train = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
y_train = dataset.iloc[:, 17:].values


dataset1 = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/sgemm_product_dataset/Regressor_test.csv')
X_test = dataset1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values
y_test = dataset1.iloc[:, 17:].values
"""
pca = PCA(n_components = 1)
NewData_train = pca.fit_transform(X_train)
NewDF_train = pd.DataFrame(data = NewData_train, columns = ['NewColumn_train1'])
print(NewDF_train)

pca1 = PCA(n_components = 1)
NewData_test = pca1.fit_transform(X_test)
NewDF_test = pd.DataFrame(NewData_test, columns = ['NewColumn_test1'])
print(NewDF_test)


X_new_train = NewDF_train.iloc[:,0:].values
X_new_test = NewDF_test.iloc[:,0:].values

Max_train = X_new_train.max()
Min_train = X_new_train.min()

X_norm_train = ((X_new_train - X_new_train.min())/(X_new_train.max() - X_new_train.min()))

Max_test = X_new_test.max()
Min_test = X_new_test.min()

X_norm_test = ((X_new_test - Min_test)/(Max_test - Min_test))


Max_train_y = y_train.max()
Min_train_y = y_train.min()

y_norm_train = ((y_train - Min_train_y)/(Max_train_y - Min_train_y))

Max_test_y = y_test.max()
Min_test_y = y_test.min()

y_norm_test = ((y_test - Min_test_y)/(Max_test_y - Min_test_y))

print(X_norm_train)
print(X_norm_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc1 = StandardScaler()
X_norm_train = sc.fit_transform(X_new_train)
X_norm_test = sc.fit_transform(X_new_test)
y_norm_train = sc.fit_transform(y_train)
y_norm_test = sc.fit_transform(y_test)

#y_final_train = np.array(y_norm_train)
#y_final_test = np.array(y_norm_test)
"""

from sklearn.svm import SVR
rgr = svm.SVR(kernel = 'poly')

rgr.fit(X_train, y_train)
#accuracy = rgr.score(X_norm_test,y_test)
#print(accuracy)
y_pred = rgr.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#y_true = [3, -0.5, 2, 7]
#y_pred = [2.5, 0.0, 2, 8]
a1 = abs(y_test[0] - y_pred[0])
a2 = abs(y_test[1] - y_pred[1])
a3 = abs(y_test[2] - y_pred[2])
a4 = abs(y_test[3] - y_pred[3])
a5 = abs(y_test[4] - y_pred[4])
a6 = abs(y_test[5] - y_pred[5])
a7 = abs(y_test[6] - y_pred[6])
a8 = abs(y_test[7] - y_pred[7])
a9 = abs(y_test[8] - y_pred[8])
a10 = abs(y_test[9] - y_pred[9])
a11 = abs(y_test[10] - y_pred[10])
a12 = abs(y_test[11] - y_pred[11])
#a13 = abs(y_test[12] - y_pred[12])
a = a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12
#print(a)
mean_a = a/12
#print(mean_a)
#print(np.mean(y_test-y_pred)**2)
absolute_error = mean_absolute_error(y_test,y_pred)
squared_error = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print('R-2 Measure is')
print(r2)
print('Absolute Error is')
print(absolute_error)
print('Squared Error is')
print(squared_error)
#print(y_test)
#print(y_pred)
#print(len(X_train))
#print(len(y_train))
"""
plt.scatter(X_norm_train,y_norm_train, color = 'red')
plt.plot(X_norm_test, rgr.predict(X_norm_test), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_norm_test), max(X_norm_test), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_norm_test, y_norm_test, color = 'red')
plt.plot(X_grid, rgr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
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
#objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(y_test))
#performance = [10,8,6,4,2,1]
 
plt.bar(test,y_test[:,0],alpha=0.3)
#plt.xticks(y_pos, objects)
plt.ylabel('Predicted/Test')
plt.title('SVR Poly')
plt.legend()

#objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(y_pred))
#performance = [10,8,6,4,2,1]
 
plt.bar(pred,y_pred,alpha=0.3)
#plt.xticks(y_pos, objects)
plt.ylabel('Predicted/Test')
plt.title('SVR Poly')
plt.legend()
plt.show()


