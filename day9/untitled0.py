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
























