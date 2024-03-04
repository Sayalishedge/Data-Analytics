# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:26:54 2023

@author: dbda
"""

import scipy.stats as stats
import matplotlib.pyplot as plt

#Define data groups
group_a=[85,88,90,92,95,87,89,91,94,86,88,92,90,93,85,89,91,87,92,90,88,93,86,89,91]
group_b=[78,82,80,85,88,81,83,79,84,87,80,82,85,78,81,83,79,82,80,85,78,81,83,79,84,87,80,82,85,78]

#Perform t-test
t_statistic, p_value = stats.ttest_ind(group_a,group_b)

#Print results
print("T-statistic: ",t_statistic)
print("P-value: ",p_value)

#Conclusion
if p_value < 0.05:
    print("Conclusion: Stastically significant difference between Group A and Group B means (p-value < 0.05) so reject H0")
else:
    print("Conclusion: No Stastically significant difference between Group A and Group B means (p-value >= 0.05) so do not rejec H0")

    
#QQ plots    
stats.probplot(group_a, dist='norm',plot=plt)
plt.title("QQ plot for Group A")
plt.show()

stats.probplot(group_b,dist='norm', plot=plt)
plt.title("QQ plot for Group B")
plt.show()


#Scatter plot having both groups
plt.figure(figsize=(10,6))
plt.scatter(range(len(group_a)),group_a,color='blue',marker='o',label='Group A')
plt.scatter(range(len(group_b)),group_b,color='red',marker='s',label='Group B')
plt.xlabel('Data Point Index')
plt.ylabel('Score')
plt.title('Comparison of Group A and Group B scores')
plt.legend()
plt.grid(True)
plt.show()


#Finding the critical value
from scipy.stats import t

df=len(group_a) + len(group_b)-2 #for unequal variances
alpha=0.05
critical_value = t.ppf(1-alpha/2,df)#two-tailed test
print(critical_value)

#Conclusion
if (t_statistic>critical_value):
    print("Conclusion: Reject H0 as t_statistic > critical_value")
else:
    print("Conclusion: Do not Reject H0 as t_statistic > critical_value")
























