# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:31:17 2023

@author: dbda
"""

from statsmodels.stats.weightstats import ztest as ztest

#enter IQ levels of 20 patient
data = [88,92,94,94,96,97,97,99,99,105,109,109,109,110,112,112,113,114,115]

#perform one sample z-test
#The func returns 2 values : the z-statistics and corresponding p-value
print(ztest(data, value=200))

