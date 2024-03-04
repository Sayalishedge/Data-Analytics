# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:23:30 2023

@author: dbda
"""

import pandas as pd
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

#Show as a scatterplot
scatter_plot = advert.plot.scatter(x='TV',y='Sales', color='r')
plt.show()

