# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 10:59:40 2023

@author: dbda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rev_m = 170
rev_stdev = 20
iterations =1000

np.random.seed(5)
#Create a normal distribution
revenue = np.random.normal(rev_m,rev_stdev,iterations)
print(revenue)
# print(np.where(revenue == min(revenue)))
# print(np.where(revenue == max(revenue)))
# print("max revenue:",max(revenue))
# print("min revenue",min(revenue))

#plot the revenue
plt.figure(figsize=(15,6))
plt.plot(revenue)
plt.title('Revenue Simulation')
plt.show()

#Now COGS(Cost of Goods Sold)
COGS = (revenue * np.random.normal(0.6,0.1))

plt.figure(figsize=(15,6))
plt.title("Cost Of Goods Sold")
plt.plot(COGS)
plt.show()

#print(COGS[324])
#print(COGS[968])

#Calculate gross profit
Gross_profit = revenue - COGS
print(Gross_profit)

plt.figure(figsize=(15,6))
plt.title('Gross Profit Simulation')
plt.plot(Gross_profit)
plt.show()

#print(np.where(Gross_profit == max(Gross_profit)))

#Create a stacked bar chart
numbers = list(range(1,1001))

plt.figure(figsize=(10,6))
plt.bar(numbers, revenue, label='Revenue', color='skyblue')
plt.bar(numbers, COGS, bottom=revenue, label='COGS', color='lightcoral')
plt.bar(numbers, Gross_profit, bottom=[r - c for r, c in zip(revenue, COGS)],label='Gross Profit', color='lightgreen')
plt.xlabel('Number of times')
plt.ylabel('Amount ($)')
plt.title('Revenue, COGS and Gross Profit using Monte Carlo')
plt.legend()
plt.show()














