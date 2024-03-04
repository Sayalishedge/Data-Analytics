# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:31:05 2023

@author: dbda
"""

import numpy as np
from scipy.stats import f_oneway

method_A = [85, 88, 91, 78, 82]
method_B = [75, 79, 80, 82, 78]
method_C = [90, 85, 88, 92, 87]



#Perform ONE WAY ANOVA
f_statistic, p_value = f_oneway(method_A, method_B, method_C)

print("F statistic: ",f_statistic)
print("P-value: ",p_value)

alpha =0.05
if p_value < alpha:
    print("Reject the null hypothesis: There are significant differences in the means")
else:
    print("Failed to reject the null hypothesis: No significant differences in the means")
    
'''
method_A = [1,2,3,4,5]
method_B = [1,2,3,4,5]
method_C = [1,2,3,4,5]

F statistic:  0.0
P-value:  1.0
Failed to reject the null hypothesis: No significant differences in the means

'''    