# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:09:02 2023

@author: dbda
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']

pima = pd.read_csv(r'F:\data_analytics\dataset\pima-indians-diabetes.csv', header=None,
                   names=col_names)
print(pima)

#Splitting the data into features and target variables
feature_col = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']
X = pima[feature_col] #Features
y = pima.label 

#Splitting the data into training and testing data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = DecisionTreeClassifier()
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(model , out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_col, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())


#Trying new data
feature_col = ['pregnant','insulin','bmi','glucose']
X = pima[feature_col] #Features
y = pima.label 

#Splitting the data into training and testing data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = DecisionTreeClassifier()
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(model , out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_col, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())













