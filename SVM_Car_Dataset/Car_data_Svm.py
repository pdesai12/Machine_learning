import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.decomposition import PCA
#start = datetime.datetime(2012,1,1)
#end = datetime.datetime(2017,1,1)

dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/1_2-19.csv')

X = dataset.iloc[:, [5,7,8,9,12,13,14,15,16,17]].values
#X = dataset.iloc[:, [5,7,8]].values
y = dataset.iloc[:, 2].values
#print(X[:,1:4])
#print(y)

#X = np.array(df.drop(['Lable'],1))
#y = np.array(df['Lable'])

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:4])
X[:, 1:4] = imputer.transform(X[:, 1:4])
#print(X[:,1:4])

#Encoding categorical data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
#Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#print(y)
#Reduce dimensionality
pca = PCA(n_components = 2)
NewData_train = pca.fit_transform(X)
NewDF_train = pd.DataFrame(data = NewData_train, columns = ['NewColumn_train1' , 'NewColumn_train2'])

#pca1 = PCA(n_components = 2)
#NewData_test = pca1.fit_transform(X_test)
#NewDF_test = pd.DataFrame(NewData_test, columns = ['NewColumn_test1' , 'NewColumn_test2'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.fit(X_train,y_train).predict(X_test)
accuracy = clf.score(X_test,y_test)
print(accuracy)
print(y_pred)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



#Design and Predict Classifier(KNN)
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p=2)
clf1.fit(X_train, y_train)

accuracy1 = clf1.score(X_test, y_test)
print(accuracy1)
y_pred1 = clf1.predict(X_test)
print(y_pred1)
