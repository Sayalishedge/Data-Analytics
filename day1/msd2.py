# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:23:07 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"F:\data_analytics\day1\MS_Dhoni_ODI_record.csv")
print(df.info())
print(df.head())

#datacleaning
df['opposition']
df['opposition'] = df['opposition'].apply(lambda x: x[2:])
df['opposition']

#Add a feature year using the match date column
df['date']
df['date'] = pd.to_datetime(df['date'],dayfirst=True)
df['year'] = df['date'].dt.year.astype(int)
print(df['year'])
print(df['date'])

#distinguish between not out and out
df['score']
df['score'] = df['score'].apply(str)
df['not_out'] = np.where(df['score'].str.endswith('*'),1,0)
df[['not_out','score']]

df['score']
df['odi_number'].value_counts()

df['sixes'].isnull()

#dropping odi_number
df.drop(columns='odi_number',inplace=True)

df.isnull().sum()
