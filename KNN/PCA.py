import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Get Data from CSV file into dataframe
dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/Ml_dataset.csv')
X_train = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
y_train = dataset.iloc[:, 10].values


dataset1 = pd.read_csv('/Users/pruthvishpatel/Downloads/ML_Project/Testing.csv')
X_test = dataset1.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
y_test = dataset1.iloc[:, 10].values


"""
dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/Ml_Project/Ml_dataset.csv')
X_train = dataset.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y_train = dataset.iloc[:, 10].values

dataset1 = pd.read_csv('/Users/pruthvishpatel/Downloads/Ml_Project/Testing.csv')
X_test = dataset1.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y_test = dataset1.iloc[:, 10].values
X_test1 = dataset1.iloc[[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9]].values
"""
#from sklearn.cross_validation import train_test_split
#X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

#Reduce dimensionality
pca = PCA(n_components = 2)
NewData_train = pca.fit_transform(X_train)
NewDF_train = pd.DataFrame(data = NewData_train, columns = ['NewColumn_train1' , 'NewColumn_train2'])

pca1 = PCA(n_components = 2)
NewData_test = pca1.fit_transform(X_test)
NewDF_test = pd.DataFrame(NewData_test, columns = ['NewColumn_test1' , 'NewColumn_test2'])

#X_t1=[[3,1,1,1,2,1,2,1,1],[5,2,4,1,1,1,1,1,1],[3,1,1,1,2,1,2,1,1],[1,1,1,1,1,1,2,1,1],[4,1,1,1,2,1,2,1,1],[5,4,6,8,4,1,8,10,1],[5,3,2,8,5,10,8,1,2],[10,5,10,3,5,8,7,8,3],[4,1,1,2,2,1,1,1,1]]
#X_t1 = X_t1.reshape(1, -1)
#pca2 = PCA(n_components = 2)
#New_T1 = pca2.fit_transform(X_test)
#print('/n')
#print(New_T1)
#NewDF_t1 = pd.DataFrame(data = New_T1, columns = ['NewColumn_test1' , 'NewColumn_test2'])

#Use New Transformed Dataset
X_New_Train = NewDF_train.iloc[:, [0,1]].values
length = len(X_New_Train)
#print(length)
#print(X_New_Train)

X_New_Test = NewDF_test.iloc[:, [0,1]].values

#X_new_t1 = NewDF_t1.iloc[:, [0,1]].values
#print(X_new_t1)
Max = X_New_Train.max()
Min = X_New_Train.min()
#Normalize Data
X_norm_train = ((X_New_Train - X_New_Train.min())/(X_New_Train.max() - X_New_Train.min()))
#print(X_norm_train)
X_norm_test = ((X_New_Test - Min)/(Max - Min))
#print(X_norm_test)
#X_norm_t1 = ((X_new_t1 - Min)/(Max - Min))
#print(X_norm_t1)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc1 = StandardScaler()
X_norm_train = sc.fit_transform(X_norm_train)
X_norm_test = sc.transform(X_norm_test)
#X_norm_t1 = sc.transform(X_norm_t1)


from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_norm_train, y_train)
y_pred = clf.fit(X_norm_train,y_train).predict(X_norm_test)
accuracy = clf.score(X_norm_test,y_test)
print(accuracy)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Design and Predict Classifier(KNN)
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p=2)
clf1.fit(X_norm_train, y_train)

accuracy1 = clf1.score(X_norm_test, y_test)
print(accuracy1)
y_pred1 = clf1.predict(X_norm_test)
print(y_pred1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)


from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()
clf2.fit(X_norm_train, y_train)

accuracy2 = clf2.score(X_norm_test, y_test)
print(accuracy2)
# Predicting the Test set results
y_pred2 = clf2.predict(X_norm_test)
print(y_pred2)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

"""
from sklearn.svm import SVR
rgr = SVR(kernel = 'poly')

rgr.fit(X_norm_train, y_train)
#accuracy = rgr.score(X_norm_test,y_test)
#print(accuracy)
y_pred = rgr.predict(X_norm_test)
print(y_pred)
"""

from matplotlib.colors import ListedColormap
X_set, y_set = X_norm_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf1.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('white', 'yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('Purple', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.legend()
#plt.show()

# Visualising the Test set results
#from matplotlib.colors import ListedColormap

X_set, y_set = X_norm_test, y_pred
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('white', 'yellow')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('black', 'blue'))(i), label = 'Train')
#plt.title('Naive Bayes (Train_Test)')
plt.legend()
plt.show()
"""

plt.scatter(X_norm_train, y_train, color = 'red')
plt.plot(X_norm_train, rgr.predict(X_norm_test), color = 'blue')
plt.title('SVR')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X_norm_train), max(X_norm_train), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_notm_train, y_train, color = 'red')
plt.plot(X_grid, rgr.predict(X_grid), color = 'blue')
plt.title('SVR')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
plt.show()
"""
