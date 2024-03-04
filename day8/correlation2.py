# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:14:01 2023

@author: dbda
"""
import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\loan.csv")
numeric_col = ['age','employ' ,'address','income', 'debtinc', 'creddebt', 'othdebt']

#Extracts a numeric data subset of the DataFrame
corr = data.loc[:,numeric_col].corr()
print(corr)

#Vizualize
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(corr, vmin=-0.5, vmax=0.8, cmap='RdYlGn')
plt.show()






