# -*- coding: utf-8 -*-
"""
Assignment (Normal Distribution)
Use datasets Men ODI Player Innings Stats -20th Century.csv and Men ODI Player Innings Stats -21st Century.csv

Consider a batsman, eg.Sachin Tendulkar
Calculate and show his Z-scores for each innings with reference to himself
(That is what is his mean innings score, and hence, what are the Z-Scores in each of his innings?)

A)Tabulate 
B)Plot the Z-Scores on a graph
"""
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt



cent_20 = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\Men ODI Player Innings Stats - 20th Century.csv")
cent_21 = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\Men ODI Player Innings Stats - 21st Century.csv")
cent_21=cent_21.drop_duplicates() #cent_21 had repeated rows thrice

df = pd.concat([cent_20,cent_21])

print(df.columns)

#selecting data for Sachin Tendulkar
df_sachin = df[df['Innings Player'] == 'SR Tendulkar']

print(df_sachin.columns)

#Remove nan and convert into int
df_no_duplicates = df_sachin.drop_duplicates(subset=['Innings Player','Innings Date'])
df_sachin = df_no_duplicates.copy()

df_sachin['Innings Runs Scored Num'] = pd.to_numeric(df_sachin['Innings Runs Scored Num'],errors='coerce')
df_sachin = df_sachin.dropna(subset=['Innings Runs Scored Num']).astype({'Innings Runs Scored Num' : 'int'})


#A)Tabulate 
mean = np.mean(df_sachin['Innings Runs Scored Num'])
sd=np.std(df_sachin['Innings Runs Scored Num'])

z_scores = (df_sachin['Innings Runs Scored Num'] - mean) / sd

df_sachin['z_scores']=z_scores

#B)Plot the Z-Scores on a graph
plt.figure(figsize=(8,6))
plt.hist(df_sachin['Innings Runs Scored Num'], bins=5, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title("Distribution of sachin's z-score")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(z_scores, bins=5, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title("Distribution of sachin's z-score")






