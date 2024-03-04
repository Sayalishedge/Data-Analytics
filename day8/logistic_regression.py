# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:17:43 2023

@author: dbda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\diabetes.csv")
print(df)
print(df.columns)

X = df.drop("Outcome", axis=1)
y=df['Outcome']



print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#print(X_train)

X_test = sc.transform(X_test)
#print(X_test)

from sklearn.linear_model import LogisticRegression
#liblinear is a solver suitable for small to medium sized datasets.
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train,y_train)

y_test_prediction = classifier.predict(X_test)
print(y_test_prediction)

comparison = pd.DataFrame({'Actual' : y_test, 'Predicted ':y_test_prediction})
print(comparison[0:10])

from sklearn.metrics import accuracy_score
#Calculates and prints the accuracy score, which measures the fraction of correctly predicted
#instances in the test set.

print(accuracy_score(y_test,y_test_prediction))

y_train_prediction = classifier.predict(X_train)
print("Accuracy Score of LR:",accuracy_score(y_train,y_train_prediction))

from sklearn.metrics import confusion_matrix
#Calculates and prints the confusion matrix for the test set
conf_mat = confusion_matrix(y_test,y_test_prediction)
print(conf_mat)

#Creates a heatmap using seaborn to visualize the confusion matrix.

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.heatmap(conf_mat,annot=True, fmt='d')
plt.title('Confusion matrix of test data')
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_prediction))


TN = conf_mat[0][0]
FP = conf_mat[0][1]
FN = conf_mat[1][0]
TP = conf_mat[1][1]

#print(TN)

recall = TP/(TP+FN)
print("Recall:",recall)
precision = TP/(TP+FP)
print("Precision: ",precision)
specificity = TN/(TN+FP)
print("Specificity: ",specificity)
accuracy = (TP+TN)/(TP+FP+FN+TN)
print("Accuracy: ",accuracy)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

model_score = accuracy_score(y_test, predictions)
print("Model Score(Accuracy using Decision Tree): ",model_score)









