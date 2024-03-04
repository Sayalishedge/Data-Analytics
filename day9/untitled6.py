# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:34:25 2023

@author: dbda

Linear programming Model for C-DAC ACTS:
    
"""
from pulp import *
 

# create the problem  variable to contain the problem data
model = LpProblem("C-DAC ACTS problem",LpMaximize)

# create 3 variables table,chairs and bookcases
# Parameters: Name,Lower limit,Upper Limit,Data type
x1 = LpVariable('teaching hour',0,None,LpInteger)
x2 = LpVariable('studentsenrolled',0,None,LpInteger)


# create maximize objective function
model += 2*100 *x1  +12 * x1

#create three constraints 
model += x1<=40*100,'faculty hour'
model += x1<=50*12,"classroom hour"
model += x2 <=70*12,'Student capacity '
model += x2 <= 500/0.5 ,'laptops'



# the problem is solved using pulp's choice of solver 
model.solve()

# Each of the variables is printed with it's resolved optimum value
for v in model.variables():
    print(v.name,'=',v.varValue)