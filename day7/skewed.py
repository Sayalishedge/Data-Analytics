# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:33:25 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#Generate a normal distribution with 10000 data points, plot it and show statistics
np.random.seed(33)
norm_sample = np.random.randn(10000)
sns.histplot(x = norm_sample, kde =True);
plt.show()

print('mean = ',np.mean(norm_sample))
print('median = ',np.median(norm_sample))
print('std = ',np.std(norm_sample))


#Generate a sample of 10000 data points from a right-skewed distribution
skewed_data_R=stats.skewnorm.rvs(10, size=10000)
sns.histplot(x= skewed_data_R, kde=True)
plt.show()

print('mean = ',np.mean(skewed_data_R))
print('median = ',np.median(skewed_data_R))
#print('std = ',np.std(skewed_data_R))


#Generate a sample of 10000 data points from a left-skewed distribution
skewed_data_L=stats.skewnorm.rvs(-10, size=10000, loc=5)
sns.histplot(x= skewed_data_L, kde=True)
plt.show()

print('mean = ',np.mean(skewed_data_L))
print('median = ',np.median(skewed_data_L))
#print('std = ',np.std(skewed_data_L))

#Q-Q plots
stats.probplot(norm_sample, plot=plt)
plt.show()
stats.probplot(skewed_data_R, plot=plt)
plt.show()
stats.probplot(skewed_data_L, plot=plt)
plt.show()

#Print the skewness of each dataset
print(stats.skew(norm_sample))
print(stats.skew(skewed_data_L))
print(stats.skew(skewed_data_R))

#Perform a skewness test on each dataset
#returns a numerical value representing the skewness of the dataset
print(stats.skewtest(norm_sample))
print(stats.skewtest(skewed_data_L))
print(stats.skewtest(skewed_data_R))

#A small p-value (typically less than 0.05) suggests that the data is significantly skewed



























