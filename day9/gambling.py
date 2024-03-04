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
    res = cal(results)
    return results,res

print(monte_carlo_simulation())
print(monte_carlo_simulation())
print(monte_carlo_simulation())

#Part 3 -- 1000 times

           