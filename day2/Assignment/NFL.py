# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:40:31 2023

@author: dbda
"""

import numpy as np
import pandas as pd

df = pd.read_csv(r"F:\data_analytics\dataset\NFL Play by Play 2009-2016 (v3).csv")
print(df.head())
print(df.info('Date'))
print(df.describe())
print(df.columns)
print(df.shape)


print(df.down.unique())
print(df.Date)
df['Date'] = pd.to_datetime(df['Date'])
df['Date'].isnull()

df2=df[df['desc'].isnull()]
print(df2['desc'])

#Desc columns nan values filled with "No description"
df['desc'].fillna("No description",inplace=True)
print(df['desc'].iloc[93350])

df['TimeSecs']

#Filled Nan values in TimeSecs with 0
df2=df[df['TimeSecs'].isnull()]
print(df2['TimeSecs'])
df['TimeSecs'].fillna(0,inplace=True)


df['yrdln']

df['yrdline100']
df['ydstogo']
df['ydsnet']

print(df['FirstDown'][df['FirstDown'].isnull()])


























