# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:49:52 2023

@author: dbda
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\loan.csv")

#Calculate the correlation coefficient
correlation_coefficient = df['age'].corr(df['income'])

print(f"Correlation coefficient : {correlation_coefficient}")

#Plot a scatter plot to visualize the data
plt.scatter(df['age'], df['income'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title(f"Scatter plot (correlation: {correlation_coefficient:.2f})")

#Identify potential outliers (eg. values with residual greater than 2 times the standard deviation)
residuals = df['age'] - df['income']
print(residuals)

std_deviation = residuals.std()
outliers = df[abs(residuals) > 2 * std_deviation]

#Highlight potential outliers on the scatter plot
plt.scatter(outliers['age'], outliers['income'], color='red', label='Outliers')
plt.legend()
plt.show()
