# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 08:46:19 2023

@author: dbda
"""

from scipy.special import comb

n = 3  #The number of trials
k = 2 #The number of success 
p = 0.70 #Success rate

p_binomial = comb(n,k)*p**k*(1-p)**(n-k)

result = p_binomial*100

print(p_binomial)
print(result)

print('__________________________________________________________________________')
'''
Binomial distribution:
    n=no.of trials

Hospital records show that of patients suffering from a specific disease, 75% die of it.
What is the prob that out of six randomly selected patients, four will die

'''
from scipy.stats import binom
probab = binom.pmf(k=4,n=6,p=0.25)
print("Probability that of six randomly selected patients, four will recover is = ",probab)


'''
A blindfolded marksman finds that on the average, he hits the target 4 times out of 5.
If he hires 4 shots, what is a prob of 
a) more than 2 hits
b) at least 3 misses    
'''

p=4/5
n=4
#for more than 2 hits, k can be either 3 or 4
#prob of more than 2 hits i.e k=3 or 4
prob_a=binom.pmf(k=3,n=n,p=p)
print('Probability of more than 2 hits = ',prob_a)

#At least 3 misses also means 0 or 1 hits, k can either be 0 or 1
#
prob_b = binom.pmf(k=1,n=n,p=p)
print('Probability of at least 3 misses',prob_b)

'''
In a random experiment, a coin is taken for tossing and it was tossed
exactly 10 times. What are the probabilities of obtaining exactly six heads 
out of total 10 tosses?
'''
from scipy.stats import binom
n=10
p=0.5
x_val=list(range(n+1))
dist = [binom.pmf(x,n,p) for x in x_val]

#to print the distribution table
print("x\tp(x)")
for i in range(n+1):
    print(str(x_val[i]) + '\t' + str(dist[i]))
    
#to get values for mean and variance
m,v = binom.stats(n, p)
print("Mean = ",str(m))
print("Variance = ",str(v))

#plotting the graph
dtable = [binom.pmf(x,n,p)for x in x_val]

import matplotlib.pyplot as plt
plt.title('Distribution plot')
plt.bar(x_val, dtable)
plt.show()

'''
Practice Question
Normally 65% of all the students who appear for C-DAC entrance test clear it.
50 students from a coaching class have appeared for C-DAC March 2024 entrance test.
What is the probability that none of them will clear it?
What is the probability that more than 40 will clear it?
'''

p=0.65
n=50
#Prob that none of them will clear it k=0
prob_a = binom.pmf(k=0,n=n,p=p)
print('Probability that none of them will clear it = ',prob_a)

#Prob that more than 40 will clear it
prob_b = binom.pmf(k=40,n=n,p=p)
print('Probability that more than 40 will clear it = ',prob_b)




















