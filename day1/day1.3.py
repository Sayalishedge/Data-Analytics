# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:57:58 2023

@author: dbda
"""
import numpy as np
#print array from 0 to 9
print(np.arange(0,10))

print(np.zeros(3))


print(np.zeros((5,5)))


print(np.ones(3))

print(np.ones((3,4)))

print(np.random.rand(2))
print(np.random.rand(5,2))

print(np.random.rand(2))


np.random.seed(42)
print(np.random.rand(2))

print(np.random.rand(5,5))

print(np.random.randn(5,5))
print("-----------------------------------------------------------------------")

arr=np.random.randint(0,100,10)
print("array is : ",arr)
condition=arr>50
result=np.where(condition)
print(result)



print("filtered values", arr[result])

#2 random arrays
arr1=np.random.randint(0,10,5)
arr2=np.random.randint(10,20,5)

condition=arr1>5
result=np.where(condition,arr1,arr2)
print(result)

print("array 1: ",arr1)
print("array 2: ",arr2)

print("combined arrays based on condition", result)

print("------------------------------------------------------------------------")





















