# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:49:53 2023

@author: dbda
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:01:13 2023

@author: dbda
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv(r"F:\data_analytics\dataset\pima-indians-diabetes.csv",names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insuin','BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'])
             
print(dataset.columns)
print(type(dataset))

#Extract feature variables in X and Labels in y
#All the features
X = dataset.iloc[:,0:-1].values

print(X)
y = dataset.iloc[:,-1].values
print(y)

#Splitting the dataset into training(80%) and testing(20%)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

'''
Datafitment
fit_transform method computes the mean and standard deviation of each feature in the training set
and then scales the features based on these statistics.
After scaling, the feature variables in X_train will have mean of 0 and standard Deviation of 1

transform method doesn't recompute the mean and sd. instead, it uses the mean and sd 
that were computed from the training set suring the fit_transformation step.
This ensures that the same scaling transformation is applied consistently to both the training and the testing sets.
'''
X_train = sc.fit_transform(X_train)
#print(X_train)

X_test = sc.transform(X_test)
print(X_test)


#Training the Naive Bayes model 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)
print(y_pred) #Our predicted values
print(y_test) #Actual values


#Making confusion matrix and calculating the accuracy of the algo
from sklearn.metrics import confusion_matrix,accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

mat = confusion_matrix(y_test, y_pred)
print(mat)


'''

All the features
0.7922077922077922
[[93 14]
 [18 29]]

o/p for BloodPressure and BMI
0.6948051948051948
[[94 13]
 [34 13]]

o/p for Pregnancy and


'''
#Test the classifier on new data
new_data = {
    'Hours Studied':[6],
    'Previous Scores':[80],
    'Extracurricular Activities':[1],
    'Sleep Hours':[7],
    'Sample Question Papers Practiced': [3]
    }

new_df = pd.DataFrame(new_data)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
new_pred = classifier.predict(new_df)

print(new_pred)















































































































































































































































































































































































































































































































































