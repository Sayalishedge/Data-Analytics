# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:50:20 2023

@author: dbda
"""
import numpy as np
import pandas as pd
df = pd.read_csv(r"F:\data_analytics\day1\MS_Dhoni_ODI_record.csv")
print(df)
print(df.info())

print(df['runs_scored'].isnull())
df1 = df[df['runs_scored'].isnull()]
print(df1)


df = df.loc[((df['score'] != 'DNB') & (df['score'] !='TDNB')),'runs_scored':]
print(df['runs_scored'])

df['runs_scored'] = df['runs_scored'].astype(int)
print(df.info())

#on which ground did he score highest runs
df1 = df[['ground','runs_scored']]
print(df1)
print(df1.sort_values(by=['runs_scored'],ascending=False))







