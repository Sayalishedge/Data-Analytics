# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:11:43 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\cdac\dataAnalytics\dataset\tips.csv")
print(df)

print(type(df))

print(df.info())

print(df.describe())

#crosstab table for tip and sex columns
ct=pd.crosstab(df['sex'], df['tip'])
print(ct)

#crosstab table for tip and sex columns 
ct_mean=pd.crosstab(df['time'], df['tip'],
                    values=df['tip'],aggfunc='mean')
print(ct_mean)

#crosstab table for smoker and day with row and column margin
import seaborn as sns
ct_margins=pd.crosstab(df['smoker'],df['day'],margins=True)
sns.heatmap(ct_margins,cmap='coolwarm',annot=True,fmt='d')
plt.show()
print(ct_margins)

#crosstab table for smoker and day
ct_norm=pd.crosstab(df['smoker'], df['day'],margins=True)
print(ct_norm)


#crosstab table for smoker and day with normalization
ct_norm=pd.crosstab(df['smoker'], df['day'],normalize=True,margins=True)
print(ct_norm)


#crosstab table for smoker and day with visuaization
ct_viz=pd.crosstab(df['smoker'], df['day'])
ct_viz.plot(kind='bar',stacked=True)
plt.show()


#crosstab table for smoker and day with visuaization
ct_viz=pd.crosstab(df['smoker'], df['time'])
ct_viz.plot(kind='bar')
plt.show()


#crosstab table for smoker and day with visuaization
ct_viz=pd.crosstab(df['smoker'], df['sex'])

ct_viz.plot(kind='bar')
plt.show()
