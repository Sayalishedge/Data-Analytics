# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:54:19 2023

@author: dbda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset('flights')
print(df.head())
#convert the date to yyyy-mm-dd format in a new column named yearMonth
df['yearMonth']="01-"+df['month'].astype(str)+"-"+df['year'].astype(str)

#yearMonth is of type object we may have problems later so convert it inot datetime

df['yearMonth']=pd.to_datetime('01-'+df['month'].astype(str)+'-'+df['year'].astype(str))

print(df.info)
print(df.head())

#make yearMonth columns as the DF index

df.set_index('yearMonth', inplace=True) # inplace will make the change permanent to the df
print(df.head())

#now plot
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x=df.index, y=df.passengers)
plt.show()

#the graph will show patterns(eg.seasonality - data going up and down)
#refer to the slides for explanations
#we see in our graph two pattern :seasonality and trend


#calculate and plot rolling mean and standard deviation for 12 months
df['rollMean']=df.passengers.rolling(window=12).mean()
df['rollStd']=df.passengers.rolling(window=12).std()

print(df['rollMean'])
print(df['rollStd'])

plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y=df.passengers)
sns.lineplot(data=df,x=df.index,y=df.rollMean)
sns.lineplot(data=df,x=df.index,y=df.rollStd)
plt.show()



#conclusion:Mean is not stationary ,SD is stationary ; so our data is not stationary




























