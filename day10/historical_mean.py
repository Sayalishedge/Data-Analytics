# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 08:18:21 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#If there are warnings due to multiple indexing operations, ignore them
pd.options.mode.chained_assignment = None

df = pd.read_csv('F:\data_analytics\dataset\jj.csv')
print(df.head())
print(df.tail())

fig, ax = plt.subplots()
ax.plot(df['date'],df['data'])
ax.set_title("Plotting of Johnson and Johnson EPS Data")
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')

#Add a vertical span(a shaded region) across the X-axis to highlight a specific range or period of interest
ax.axvspan(80,83, color='#808080', alpha=0.2)
#ax.axvspan(70,73, color='#808080', alpha=0.2)
plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966,1968, 1970, 1972, 1974, 1976, 1978, 1980])

fig.autofmt_xdate() #Automatically format and adjust x-axis labels for better redability
plt.tight_layout() #Auto adjust the spacing between subplots on elements in a figure for fitting the contents neatly.
plt.show()


#Split to train/test
train = df[:-4] #Exclude last 4 records
test = df[-4:] #Include last 4 records only

#Use mean of entire training data as the predictor
historical_mean = np.mean(train['data'])
print(historical_mean)

test.loc[:,'pred_mean'] = historical_mean #Add a new column to the DF
print(test)

'''
MAPE: Mean Absolute Percentage Error
also called as Mean Absolute Percentage Deviation(MAPD)
lower the MAPE the better and the  MAPE of 0% would mean that the model's predictions are perfectly accurate.
Performed between two arrays or lists of values, typically used for assessing the accuracy of predictive 
models, such as forecasting models.
y_true = This is an array or list of true (observed) values. These are the actual values that you want to compare
your predictions to
model.
y_pred = This is an array ofrlist of predictied values. These are the values generated by your predictive model
After each prediction, we are displaying the MAPE value -note where it is the smallest -that is our best prediction.


'''

def mape(y_true,y_pred):
    return np.mean(np.abs((y_true-y_pred) / y_true)) * 100

mape_hist_mean = mape(test['data'],test['pred_mean'])
print(mape_hist_mean)

fig, ax = plt.subplots()
ax.plot(train['date'], train['data'], 'g-.', label='Train')
ax.plot(test['date'], test['data'], 'b-',label = 'Test')
ax.plot(test['date'],test['pred_mean'], 'r--', label='Predicted')
ax.set_title("Using Historical Mean of the Entire Training Data as the Predicted Mean")
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per Share (USD)')
ax.axvspan(80, 83, color='#808080', alpha=0.2)
ax.legend(loc=2)
plt.xticks(np.arange(0,85,8), [1960, 1962, 1964, 1966,1968, 1970, 1972, 1974, 1976, 1978, 1980])
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

#looks very bad

#predict last year mean (i.e. for 1980) using only the last year of training data
#remember our training data set contains data up to year 1979 only , so it will be the mean for 1979
last_year_mean=np.mean(train['data'][-4:])
print(last_year_mean)

#now add this last year's predicted mean to the training data 
test.loc[:, 'pred_last_yr_mean']=last_year_mean
print(test)

mape_last_year_mean=mape(test['data'],test['pred_last_yr_mean'])
print(mape_last_year_mean)
#better than earlier ,but still quite poorer


fig,ax=plt.subplots()

ax.plot(train['date'], train['data'],'g-',label='Train')
ax.plot(test['date'], test['data'],'b-',label='Test')
ax.plot(test['date'],test['pred_last_yr_mean'], 'r--',label='Predicted')

ax.set_title("LAST YEAR MEAN of the training data used as the predictor ")
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per Share (USD)')
ax.axvspan(80, 83, color='#808080', alpha=0.2)
ax.legend(loc=2)
plt.xticks(np.arange(0,85,8), [1960, 1962, 1964, 1966,1968, 1970, 1972, 1974, 1976, 1978, 1980])
fig.autofmt_xdate()
plt.tight_layout()
plt.show()




























































test.loc[:, 'pred_mean'] = historical_mean #Add a new column to the DF
print(test)








