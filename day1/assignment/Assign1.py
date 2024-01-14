# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:49:15 2023

@author: dbda
"""
print("--------Q1----------------------------")
import numpy as np
np.random.seed(50)
marks=np.random.randint(0,100,10)
print(marks)

max_marks=np.max(marks)
print(max_marks)


min_marks=np.min(marks)
print(min_marks)

avg_marks=np.mean(marks)
print(avg_marks)

print("---------Q2------------------------------")

import random
import string
print(''.join(random.choices(string.ascii_lowercase,k=5)))


names=np.array(['aaa','bbb','ccc','ddd','eee','fff','ggg','hhh','iii','jjj'])
print(names)

res = dict(zip(names, marks))
print(res)
print(type(res))


m=max(res)
print(m)

Key_max = max(res, key = lambda x: res[x])  
print(Key_max)


Key_min = min(res, key = lambda x: res[x])  
print(Key_min)

avg = sum(res.values()) / len(res)
print(avg) 


arr=np.array(res)
print(type(arr))
print(arr)
condition=marks>avg
result=np.where(condition)
print(result) #got index number


print("---------Q3------------------------------")

a= np.stack((names,marks))
print(a)