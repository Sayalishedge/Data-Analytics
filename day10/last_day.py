# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 17:09:33 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time

student_data=pd.read_csv("F:\data_analytics\dataset\student-mat.csv")
print(student_data.describe())

col_str=student_data.columns[student_data.dtypes == object]
print(col_str)

#convert each catefgory value into a new column and  assign a 1 or 0 (True/False)
#value to the comon This has the benefit of not weighting a value improperl.
#simplest method is using pandas  .get_dummies() method 
#drop_first = True reduces extra column  creation (e.g. coin toss, is_head and is_tail :both arenot needed)

student_data=pd.get_dummies(student_data, columns=col_str, drop_first=True)
print(student_data.info())
print(student_data[["G1","G2","G3"]].corr())

#since  G1,G2,G3 have very high correlation we candrop G1,G2
student_data.drop(axis=1,labels=["G1","G2"])

#drop the G3 column, because we want to predict it now 
label=student_data["G3"].values
predictors=student_data.drop(axis=1,labels=["G3"]).values
print(student_data.shape)



#using PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=len(student_data.columns)-1)
pca.fit(predictors)
variance_ratio=pca.explained_variance_ratio_
print(pca.explained_variance_.shape)


#now plot
import matplotlib.pyplot as plt
#find cummulative variance, adding one independent variable at a time 
variance_ratio_cum_sum=np.cumsum(variance_ratio)
print(variance_ratio_cum_sum)

plt.plot(variance_ratio_cum_sum)
plt.xlabel('Number of components')
plt.ylabel('cummulative explained variance')
#annotate 90% variance explained by the first 6 variables only
plt.annotate('6',xy=(6,.90))
plt.show()


plt.figure(figsize=(10,5))
plt.bar(range(41),pca.explained_variance_,alpha=0.5,label='individual explained variance')
plt.ylabel('explained variance ratio')
plt.xlabel('principal components')
plt.legend(oc='best')
plt.show()


import seaborn as sns
correlation=pd.DataFrame(predictors).corr()
sns.heatmap(correlation, vmax=1,square=True,cmap='Greens')
plt.title('correlation between different features')
plt.show()

#looking at the above plot we are taking 6 variable 
pca=PCA(n_components=6)
pca.fit(predictors)
Transformed_vector=pca.fit_transform(predictors)
print(Transformed_vector)

#correlation of the 6 variable after transforming the data with PCA is 0
import seaborn as sns
correlation=pd.DataFrame(Transformed_vector).corr()
sns.heatmap(correlation, vmax=1,square=True,cmap='viridis')
plt.title('correlation between different features')
plt.show()


#verify statistically
lr=linear_model.LinearRegression()
#returns an array of scores of the estimator for each run of the cross validation
lr_score=cross_val_score(lr, predictors, label, cv=5) #five runs, 5 means
print("LR Model cross Validation score: ", + str(lr_score))
print("LR Model cross Validation Mean score: ", + str(lr_score.mean()))

#check the performance with 6 variables
lr_pca=linear_model.LinearRegression()
lr_pca_score=cross_val_score(lr_pca, Transformed_vector,label,cv=5)
print("PCA Model cross Validation score: ", + str(lr_pca_score))
print("PCA Model cross Validation Mean score: ", + str(lr_pca_score.mean()))

# we see values similar to the earlier case when we ahd 40 independent 

























