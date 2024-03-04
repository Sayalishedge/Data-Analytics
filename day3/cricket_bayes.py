# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:32:17 2023

@author: dbda
"""

import pandas as pd
import numpy as np
np.random.seed(34)
import matplotlib.pyplot as plt

toss=np.array(["won","lost"])
toss_indices=np.random.choice(range(len(toss)),size=1000)
print(toss_indices)

result=np.array(["won","lost"])
result_indices=np.random.choice(range(len(result)),size=1000)
print(result_indices)


data={"toss": toss[toss_indices],"result": result[result_indices]}
print(data)

print(toss)
print(result)

df = pd.DataFrame(data)
print(df)

p_toss_won = len(df[df['toss']=='won'])/len(df)
print(p_toss_won)

#p_toss_lost = len(df[df['toss']=='lost'])/len(df)
#print(p_toss_lost)

p_match_won = len(df[df['result']=='won'])/len(df)
print(p_match_won)

#p_match_lost = len(df[df['result']=='lost'])/len(df)
#print(p_match_lost)


#probability of winning the match when the toss was won
#A=winning the match
#B=toss was won
p_a = p_match_won
p_b = p_toss_won

p_a_and_b = len(df[(df['toss']=='won')& (df['result']=='won')])/len(df)

p_b_given_a = p_a_and_b / p_a
print(p_b_given_a)

p_a_given_b = (p_b_given_a*p_a)/p_b
print("probability of winning the match when the toss was won = ",p_a_given_b)


