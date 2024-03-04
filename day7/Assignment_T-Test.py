# -*- coding: utf-8 -*-
"""
Assignment ( Two-Sample/Independent T-Test)
Use the dataset women_shoe_prices.csv

Use the 'Colors column for analysis

Do pink-color shoes cost more than other color shoes?
(Note: look for pink as the only or unique color, ignore mixed/multiple colors for comparison)

2.In the cricket example, do SR Tendulkar and V Kohli have a similar batting performance?
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r"F:\data_analytics\dataset\women_shoe_prices.csv")
df.columns

df['prices.amountMax']
df.colors.unique()

pink = df[df['colors']=='Pink']
other = df[df['colors'] != 'Pink']

print(pink.colors)
print(pink['prices.amountMax'].unique())
print(other.colors)
pink = pink.dropna(subset=['colors'])
other = other.dropna(subset=['colors'])
print(other.colors)

pink_cost = pink['prices.amountMax']
other_cost = other['prices.amountMax']

#Perform t-test
import scipy.stats as stats
import matplotlib.pyplot as plt
t_statistic, p_value = stats.ttest_ind(pink_cost,other_cost)

#Print results
print("T-statistic: ",t_statistic)
print("P-value: ",p_value)

#Conclusion
if p_value < 0.05:
    print("Conclusion: Stastically significant difference between Pink shoes and other shoes means (p-value < 0.05) so reject H0")
else:
    print("Conclusion: No Stastically significant difference between Pink shoes and other shoes means (p-value >= 0.05) so do not reject H0")
    
#Finding the critical value
from scipy.stats import t

cv=len(pink_cost) + len(other_cost)-2 #for unequal variances
alpha=0.05
critical_value = t.ppf(1-alpha/2,cv)#two-tailed test
print("Critical value:   ",critical_value)

#Conclusion
if (t_statistic>critical_value):
    print("Conclusion: Reject H0 as t_statistic > critical_value")
else:
    print("Conclusion: Do not Reject H0 as t_statistic < critical_value")
    
    
#%%

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
df_virat = df[df['Innings Player'] == 'VP Kohli']

print(df_sachin.columns)
print(df_virat.columns)
#Remove nan and convert into int
df_no_duplicates = df_sachin.drop_duplicates(subset=['Innings Player','Innings Date'])
df_sachin = df_no_duplicates.copy()

df_sachin['Innings Runs Scored Num'] = pd.to_numeric(df_sachin['Innings Runs Scored Num'],errors='coerce')
df_sachin = df_sachin.dropna(subset=['Innings Runs Scored Num']).astype({'Innings Runs Scored Num' : 'int'})
 
df_virat['Innings Runs Scored Num'] = pd.to_numeric(df_virat['Innings Runs Scored Num'],errors='coerce')
df_virat = df_virat.dropna(subset=['Innings Runs Scored Num']).astype({'Innings Runs Scored Num' : 'int'})
    

















    