# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:32:20 2023

@author: dbda
"""

from numpy import random
random.seed(0)

#Dict to store total number of people in each age group
totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}

#Dict to store purchases made by people in each age group
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases = 0

for _ in range(100000):
    ageDecade = random.choice([20,30,40,50,60,70])
    totals[ageDecade] +=1
    purchaseProbability = float(ageDecade)/100.0
    if(random.random() < purchaseProbability):
        totalPurchases +=1
        purchases[ageDecade] +=1
        
       
print(totals)
print(purchases)        
print(totalPurchases)

#probability of purchasing at age 30
PEF = float(purchases[30]) / float(totals[30])
print('P(purchase | 30s):'+str(PEF))
        
#probability of being 30
PF = float(totals[30]) / 100000.0
print('P(30s):'+str(PF))

PE = float(totalPurchases) / 100000.0
print("P(Purchase):" + str(PE))