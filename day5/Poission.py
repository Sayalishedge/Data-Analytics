# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:43:20 2023

@author: dbda
"""

#A food delivery joint typically receives the 8 deliveries between 4 to 5 pm on friday . Whats the probablity that it will get 4 deliverys on coming friday

import math 
Lambda = 8
k= 4

p_4 = (math.e**- Lambda) * (Lambda**k)/math.factorial(k)

print("The Probablity of 4 orders: {:.3f}".format(p_4))
print("In Percentage: " + str((p_4*100)))

#Using Library function

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

mu = 8
x = 4
print(poisson.pmf(x,mu))

# Now an array
mu = 500
x =  np.arange(400, 600, 0.5)
plt.plot(x, poisson.pmf(x, mu))
plt.show()

#%%
# sealesman sells on average 3 life insuranec policies per week
# Find Probablity of:
# 1. in a given week he will sell some policy
# 2. in a given week he will sell 2 or more policies but not more than 5
# 3. Assuimg that pr week, there are 5 working days, whats probablity the on given day he will sell 1 policy


mu=3

#poission.pmf(k=0, mu=mu)

# Probablity of selling some policy in a week 
x=0
no_policy = poisson.pmf(x,mu)
probab_some = 1-no_policy
print("Probablity of selling some policy in a week:", probab_some)

# Probablity of selling 3 policy in a week 
print("Probablity of selling 3 policies in a week:" ,poisson.pmf(k=3, mu=mu))


#2 or more policies but not more than 5
print("Probablity of selling 2 or more but not more than 5:" , sum(poisson.pmf(k=[2,3,4],mu=mu)))

# Assuimg that pr week, there are 5 working days, whats probablity that on given day he will sell 1 policy
day1 = poisson.pmf(k=1, mu=3/5) #3/5 to get avg for 1 day
print("Probablity the on given day he will sell 1 policy:",day1)










