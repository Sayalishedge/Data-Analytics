# -*- coding: utf-8 -*-
"""
Here we use stats.chisquare
Primarily used for conducting a chi-square goodness-of-fit test, which compares ovserved frequencies with
expected frequencies under a specified theoretical distribution
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

#fake demographic data for U.S and Minnesota state
national = pd.DataFrame(['White']*100000 + ['hispanic']*60000 + ['black']*50000 + ['asian']*15000 + ['other']*35000)
minnesota = pd.DataFrame(['White']*600 + ['hispanic']*300 + ['black']*250 + ['asian']*75 + ['other']*150)

print(national)
print(minnesota)

#Create frequency tables (crosstabs) for both datasets using pd.crosstab
national_table = pd.crosstab(index=national[0], columns='count')
minnesota_table = pd.crosstab(index=minnesota[0], columns='count')

print("National \n",national_table)
print()
print("Minnesota \n",minnesota_table)

observed = minnesota_table
national_ratios = national_table/len(national) #Get population ratios

#calculate the expected counts by multiplying the population ratios in the national dataset by the 
#total number of obsservations in the Minnesota dataset
print(national_ratios)

expected = national_ratios * len(minnesota) #Get expected counts

#Calculate the chi-squared statistic by comparing the observed and expected xounts and summing the squared
#differences, normalized by the expected counts.
chi_squared_stat = (((observed-expected)**2)/expected).sum()

print("Calculated observed value")
print(chi_squared_stat)

#Find the critical value for a 95% confidence level with 4 degrees of freedom
# Find the critical value for 95% df = number of variable categories -1 
crit = stats.chi2.ppf(q = 0.95,df = 4)

print("Critical value")
print(crit)

#Calculate the p-value based on the chi-squared statistic and degrees of freedom.
p_value = 1- stats.chi2.cdf(x=chi_squared_stat, df = 4)
print("P value")
print(p_value)










































