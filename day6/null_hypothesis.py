# -*- coding: utf-8 -*-
"""

"""
import scipy.stats as stats

sugar_levels = [125,130,118,122,128,115,130,135,120,125,113,112,140,134121,135]
hypothesized_mean =120

#Perform one-sample t-test
t_satistic, p_value = stats.ttest_1samp(sugar_levels,hypothesized_mean)

#Compare p-value to significance level (eg.,0.05)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The average sugar level is significantly different.")
else:
    print("Fail to reject the null hypothesis. No significant difference in average sugar level")
    
    