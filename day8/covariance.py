# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:56:55 2023

@author: dbda
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

advert = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\advertising.csv")
advert.head()

#Find correlation
print(advert.TV.corr(advert.Sales))
#print(advert['TV'].corr(advert['Sales']))

print(advert.TV.corr(advert.Radio))

#Show as a heatmap, annot means show numbers in cells
dataplot = sns.heatmap(advert.corr(), cmap="YlGnBu", annot=True)
plt.show()

#Find covariance
print(advert.TV.cov(advert.Sales))
#print(advert['TV'].cov(advert['Sales']))

print(advert.TV.cov(advert.Radio))

#Covariance matrix
print(advert.cov())

#fmt='g' means use general format for numbers, not scientific notations
dataplot = sns.heatmap(advert.cov(), annot=True, fmt='g')
plt.show()


