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

test.loc[:, 'pred_mean'] = historical_mean #Add a new column to the DF
print(test)









