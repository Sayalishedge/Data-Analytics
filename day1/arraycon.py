# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:49:31 2023

@author: dbda
"""
import numpy as np

mylist=[1,2,3,4,True,3.4]
arr1=np.array(mylist)
print(type(mylist))
print(type(arr1))
print("array is",arr1)
print("list is ",mylist)

print(arr1.size)
print(arr1.itemsize)
print(arr1.ndim)
print(arr1.shape)
print(arr1.dtype.name)
print(arr1.size)
print(np.array([True,10,'abc',5.5]).dtype.name)
print("------------------------------------------------------------------")


#2d list
my_matrix=([[1,2,3],[4,5,6],[7,8,9]])
print(my_matrix)
print(np.array(my_matrix))
print()
first=np.array([[1,2,3],[4,5,6]])
second=np.array([[7,8,9],[10,11,12]])

print("first ",first)
print()
print("second ",second)
print()
print(first*2)
print(first/2)

print(first.shape)
print(second.shape)

print(first+second)

print(np.concatenate((first, second)))

print(np.concatenate((first, second),axis=1))

#concatenate 2 single arrays
print(np.concatenate((np.array([1,2,3]),np.array([4,5,6]))))

print(np.concatenate((np.array([1,2,3]),np.array([4,5,6])),axis=0))

print("-----------------------------------------------------------------------")





