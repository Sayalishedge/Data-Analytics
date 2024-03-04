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



cent_20 = pd.read_csv(r"F:\data_analytics\dataset\Men ODI Player Innings Stats - 20th Century.csv")
cent_21 = pd.read_csv(r"F:\data_analytics\dataset\Men ODI Player Innings Stats - 21st Century.csv")
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


"""
For the same data:
    Select a calendar year (eg.2015)
    Select a player who you know was among the top run-getters in that year (aggregate or total runs in the year)
    Identify the top 25 players based on their Z-scores. Display their names, total runs and Z-score in a table
    Show this in a bar plot
    Show the average Z-score for these 25 players on the plot with a horizontal line.
    Annotate the selected player(i.e show his Z-score in a different color)
    
"""
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt



cent_20 = pd.read_csv(r"F:\data_analytics\dataset\Men ODI Player Innings Stats - 20th Century.csv")
cent_21 = pd.read_csv(r"F:\data_analytics\dataset\Men ODI Player Innings Stats - 21st Century.csv")
cent_21=cent_21.drop_duplicates() #cent_21 had repeated rows thrice

df = pd.concat([cent_20,cent_21])

print(df.columns)
print(type(df))
print(df['Innings Date'])

df_year = pd.to_datetime(df['Innings Date']).dt.year
df['year'] = df_year

df_2015 = df[df['year']==2015]

grouped = pd.concat([df_2015,df['Innings Runs Scored Num']])
grouped = grouped.dropna()
grouped['year']
grouped['Innings Runs Scored Num']


df_group = df_2015.groupby('Innings Player').mean()
final_df = df_group.mean()

#To calculate z-scores
from scipy import stats
stats.zscore()





















