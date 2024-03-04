# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:13:31 2023

@author: dbda
 R1 : labor: 2*x1 +1* x2 + 2.5*x3 <=60
 R2 : Machine:0.8*x1 +0.6*x2 + 1*x3 <=16 hours
 R3 : Wood : 30*x1 + 20*x2 + 30*x3 <=400 booard-feet
 products : Chairs ($30 to profit), Tables($40),bookcases($45)
"""

from pulp import *

# create the problem  variable to contain the problem data
model = LpProblem("FurnitureProble",LpMaximize)

# create 3 variables table,chairs and bookcases
# Parameters: Name,Lower limit,Upper Limit,Data type
x1 = LpVariable('table',0,None,LpInteger)
x2 = LpVariable('chairs',0,None,LpInteger)
x3 = LpVariable('bookcases',0,None,LpInteger)

# create maximize objective function
model +=40 * x1 +30 * x2 + 45 * x3

#create three constraints 
model += 2* x1 +1*x2 +2.5 * x3 <=60,'Labour'
model += 0.8*x1 + 0.6*x2 + 1.0*x3 <=16,"Machine"
model += 30*x1 + 20*x2 +30*x3 <=400,'wood'

# Try commenting the statement below and see the difference 
model += x1>= 3,'tables'
model += x3 >= 1,'bookcases'

# the problem is solved using pulp's choice of solver 
model.solve()

# Each of the variables is printed with it's resolved optimum value
for v in model.variables():
    print(v.name,'=',v.varValue)
    