# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:56:31 2023

@author: dbda
"""

import numpy as np

def roll_dice():
    return np.sum(np.random.randint(1,7,2))

print(roll_dice())


#2-3 times
def cal(result):
    ans= result[0]*5-result[1]
    return ans
def monte_carlo_simulation(runs=1000):
    results = np.zeros(2)
    for _ in range(runs):
        if roll_dice() ==7:
            results[0]+=1
        else:
            results[1]+=1
    # res = cal(results)
    return results#,res

print(monte_carlo_simulation())
print(monte_carlo_simulation())
print(monte_carlo_simulation())

#Part 3 -- 1000 times
# takes some time to execute
results  = np.zeros(1000)
for i in range(1000):
    results[i] = monte_carlo_simulation()[0]
print(results)    

import matplotlib.pyplot as plt
ax = plt.subplot()
ax.hist(results,bins=15)
plt.show()

print(results.mean())   #General mean 
print(results.mean()*5)     # what we will get as win on an average
print(1000-results.mean())  #what we will pay on an average
print(results.mean()/1000)  # probablility of the 'we will pay' result

#the last probablility shoud be cloase to the theoretical probabilityof geting 7 when we thow two dice

           