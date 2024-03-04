# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:50:34 2023

@author: Sayali and Ratan
Assignement
Design a simple geame:
    
    Choose a number between 1 to 100
    If the sum of the digits of your chosen number exceeds 12, you win 3 dollars
    if the sum fails below the threshold, you lose 1 dollar
    the game will be repeated 1000 times to simulate a large number of trials..
    
    write code and then
    determine if 3 dollars and 1 dollar for win and lose are correct,or they should be changed.
    
"""
import numpy as np
import functools

def guess_game():
    guess = np.random.randint(1,100,1)
    return guess

# guess = guess_game()
# print(str(guess_game())[1:-1])
# print(guess)
# num = functools.reduce(lambda x,y:int(x)+int(y), str(guess)[1:-1])
# print(type(num))
def casino_owner(runs=1000):
    results = np.zeros(2)
    for _ in range(runs):
        if  int(functools.reduce(lambda x,y:int(x)+int(y), str(guess_game())[1:-1]))>12:
            results[0]+=1
        else:
            results[1]+=1
    # res = results[0]-results[1]
    return results#,res
# print(casino_owner())

#Part 3 -- 1000 times
# takes some time to execute
results  = np.zeros(1000)
for i in range(1000):
    results[i] = casino_owner()[0]
print(results)    

import matplotlib.pyplot as plt
ax = plt.subplot()
ax.hist(results,bins=15)
plt.show()

print('General mean',results.mean())   #General mean 
print('What we will get as win on an average: ',results.mean()*3)     # what we will get as win on an average
print('what we will pay on average: ',1000-results.mean())  #what we will pay on an average
print('Pobability of the \'we will pay\' result',results.mean()/1000)  # probablility of the 'we will pay' result

#th