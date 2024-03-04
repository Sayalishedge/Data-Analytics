# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:13:32 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

df = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\Banking_churn_prediction.csv")
print(df.head())

#Calculate skewness and kurtosis
skewness = skew(df['current_month_balance'])
kurt = kurtosis(df['current_month_balance'])

plt.figure(figsize=(18,6))
x_range = (0, 100000)
plt.subplot(1,3,1)
plt.hist(df['current_month_balance'],bins =10, range=x_range,color='skyblue', edgecolor='black') #Increase the number of bins
plt.title('Distribution of current month balance')
plt.ticklabel_format(style='plain', axis='both') #Disable scientific notation on x axis
plt.xticks(rotation=45)

#Plot skewness and kurtosis
plt.subplot(1,3,2)
plt.text(0.5, 0.9, f'Skewness: {skewness:.2f}', fontsize=12, ha='center',va='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.8, f'Kurtosis: {kurt:.2f}', fontsize=12, ha='center',va='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.7, f'Interpretation:', fontsize=12, ha='center',va='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.6, f'Skewness > 0 indicates right skewness', fontsize=10, ha='center',va='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.5, f'Kurtosis > 3 indicates leptokurtic', fontsize=10, ha='center',va='center', transform=plt.gca().transAxes)
plt.axis('off')

#Plot boxplot for outliers
plt.subplot(1,3,3)
plt.boxplot(df['current_month_balance'],vert=False)
plt.title('Boxplot for outliers')
plt.tight_layout()
plt.show()















