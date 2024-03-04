# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:04:51 2023

@author: dbda
"""

from statsmodels.stats .weightstats import ztest
#This is our sample
data = [88,92,94,96,97,97,99,99,105,109,109,109,110,112,112,113,114,115]


hypothesized_population_mean =110

#Put hypothesized population mean value here
z_statistic,p_value = ztest(data, value=hypothesized_population_mean)

#If the ovserved sample mean is less than the hypothesized mean, the z-statistic will be negative because the
#observed mean is below the expected mean. Else it will be positive
print(f"Z=Statistic: {z_statistic}, p-value: {p_value}")

#Now vizualize the original data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Fit a normal distribution to the data
mu, std = norm.fit(data)

#Plot the histogram
plt.hist(data,bins=10, density=True, alpha=0.6, color='g')

#Plot the PDF of the fitted normal distribution
xmin,xmax =plt.xlim()
x = np.linspace(xmin,xmax,100)
p = norm.pdf(x,mu,std)
plt.plot(x,p,'k',linewidth=2)
plt.xlabel('IQ Levels')
plt.ylabel('Probability Density')
plt.title('Fit results: mu =%.2f, std =%.2f' % (mu,std))
plt.show()

#Show hypothesized mean
ecdf = np.arange(1, len(data) + 1) / len(data)

percentile_at_test_value = np.interp(hypothesized_population_mean,np.sort(data),ecdf)

plt.step(np.sort(data),ecdf, label='Empirical CDF', where='post')
plt.axvline(hypothesized_population_mean, color='red', linestyle='dashed',linewidth=2, label='Test Value')
plt.xlabel('IQ Levels')
plt.ylabel('Cummulative Probability')
plt.title('Empirical Cumulative Distribution Function(ECDF) of IQ Levels')
plt.legend()
plt.show()















