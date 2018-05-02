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


dataset = pd.read_csv('/Users/pruthvishpatel/Downloads/Ml_Project/Ml_dataset.csv')
X_train = dataset.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y_train = dataset.iloc[:, 10].values

dataset1 = pd.read_csv('/Users/pruthvishpatel/Downloads/Ml_Project/Testing.csv')
X_test = dataset1.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y_test = dataset1.iloc[:, 10].values

#Queries 
X_test_query = [[4,1,1,3,2,1,1,1,1],[4,6,6,5,7,6,7,7,3]]
y_test_result = [[2],[4]]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = DecisionTreeClassifier( random_state = 0)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(accuracy)


y_pred = classifier.predict(X_test_query)
print("predicted Results")
print(y_pred)
print("Actual reuslts")
print(y_test_result)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

features = ["Clump Thickness" ,"Uniformity Of size", "Uniformity of Shape" ,"Marginal Adhesion" ,"Single epithelial cell size" ,"Bare Nuclei" ,"Bland Chromatin" ,"Normal Nucleoli" ,"Mitoses"]

from sklearn import tree
from io import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier,out_file=out,feature_names = features)
pydotplus.graph_from_dot_data(out.getvalue()).write_png('/Users/pruthvishpatel/Downloads/Ml_Project/En_Decision_Tree.png')
img = misc.imread('/Users/pruthvishpatel/Downloads/Decision_Tree_Classification/dt2.png')

