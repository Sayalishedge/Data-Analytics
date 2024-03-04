# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:47:12 2023

@author: dbda
"""

import scipy.stats as stats
import matplotlib.pyplot as plt

marks = [37,89,12,56,90,92,45,56,67,23,81,85,72,97,18,10]

#Normalize test with Shapiro-Wilk test
shapiro_test = stats.shapiro(marks)
print("Shapiro-Wilk p-value:", shapiro_test.pvalue)

# Q-Q plot
stats.probplot(marks, fit=stats.norm, plot=plt)
plt.xlabel("Therotical Quantities")
plt.ylabel("Sample Quantities")
plt.title("Q-Q Plot for Exam Marks")
plt.grid(True)
plt.show()


#Interpretation
if shapiro_test.pvalue < 0.05:
    print("The data likely does not follow a normal distribution.")
else:
    print("The data may be normally distributed, but the Q-Q plot can provide further insight.")
    
#Analyze the Q-Q plot:

'''
=>If the points roughly fall along the straight line, it suggest Normality

=>Deviation from the line, especially at the tails, indicate Non-normality

'''