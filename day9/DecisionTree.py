# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 08:07:40 2023

@author: dbda
"""

import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"F:\data_analytics\dataset\vehicles.csv")
print(df.head())
print(df.info())

print(df['Favorite Transport'].unique())
print(df.value_counts('Favorite Transport'))
print(df.isnull().sum())
df['Income'] = df['Income'].fillna(0.0)
print(df.head(8))

label_encoder = LabelEncoder()
encoded_genders = label_encoder.fit_transform(df['Gender'])
print(encoded_genders)

df['Gender'] = encoded_genders
print(df.head(8))
print(df.dtypes)

X = df.drop(columns='Favorite Transport')
y = df['Favorite Transport']

print(X.head())
print(y.head())

model = DecisionTreeClassifier()
model.fit(X,y)

test_df = pd.DataFrame({
    'Age':[12,30,75],
    'Gender' :[0,0,1],
    'Income': [0.0,4000,50000]
    })
print(test_df)
model.predict(test_df)


#Export to DOT file
tree.export_graphviz(model, out_file='decision_tree_model.dot', 
                     filled=True, feature_names=['Age','Gender','Income'],
                     class_names=sorted(y.unique()))

#to convert : dot decision_tree_model.dot -Tpng -o decision_tree_model.png
'''
import graphviz
from graphviz import Source
def convert_dot_to_png(dot_file_path,png_file_path):
    with open(dot_file_path,'r')as dot_file:
        dot_source = dot_file.read()
    graph = Source(dot_source,format='png')    
    graph.render(filename=png_file_path,cleanup=True, format='png', engine='dot')
convert_dot_to_png('decision_tree_model.dot', 'decision_tree_model')
'''



sns.countplot(x=df['Gender'], hue=df['Favorite Transport'])
plt.show()

sns.histplot(x=df['Income'], hue=df['Favorite Transport'])
plt.show()

#Evaluate Accuracy of the Model
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
print("Train shape: ",X_train.shape)
print("Test shape",X_test.shape)
print("Source data shape", X.shape)
print("Test input data", X_test)
print("Test target data", y_test)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

model_score = accuracy_score(y_test, predictions)
print("Model Score(Accuracy using Decision Tree): ",model_score)


#Try doing the same classification using Logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

predict_lr = classifier.predict(X_test)
print(predict_lr)
accuracy_lr = accuracy_score(y_test, predict_lr)
print("Accuracy using Logistic Regression: ",accuracy_lr)




































