# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 08:50:09 2023

@author: dbda
"""
from sklearn.datasets import fetch_california_housing

'''
return_X_y=True: This parameter specifies that you want the dataset to be split into features (in this case,
housing) and target values(in this case, target) and returned as separate variables. When retrun_X_y is set
to true, housing will contain the feature data, and target will contain the target values.
The feature variables and target values are already decided for this dataset.
Please refer to this : https://scikit-learn.org/stable/datasets/real_worls.html#california-housing-dataset
'''
housing, target = fetch_california_housing(as_frame=True, return_X_y=True)
print(housing.head())
print(target.head())

#correlation of all the columns with each other
print(housing.corr())

#Corretlation of two specific columns
corr = housing.corr()
print(corr['MedInc']['AveRooms'])

#Visual view
import seaborn as sns
import matplotlib.pyplot as plt

cmap = sns.diverging_palette(10,220, as_cmap=True)#(10 t 220 is the range of the color palette)
sns.heatmap(corr,vmin=-1.0, vmax=1.0, square=True, cmap=cmap)



plt.show()

