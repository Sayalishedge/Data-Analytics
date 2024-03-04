# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:57:44 2023

@author: dbda
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

tips=sns.load_dataset("tips")

print(tips) 

ax = sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()