
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv(r"F:\data_analytics\dataset\Social_Network_Ads.csv")

#Extract feature variables in X and Labels in y
X = dataset.iloc[:,[2,3]].values

from collections import Counter

#get gender frequencies
gender_counts=Counter(dataset['Gender'])

#define a function to encode gender values
def encode_gender(gender):
    return gender_counts[gender]/len(dataset)


#Apply the encoding function to the gender column
dataset['Gender']= dataset['Gender'].apply(encode_gender)

#update the feature with the encoded gender column
X=dataset.iloc[:,[1,2,3]].values
print(X)
print(dataset.columns)


y = dataset.iloc[:,-1].values

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
output with out encoding
0.9125
[[55  3]
 [ 4 18]]

o/p with frequency encoding
0.925
[[56  2]
 [ 4 18]]
'''



















































































































































































































































































































































































































































































































































