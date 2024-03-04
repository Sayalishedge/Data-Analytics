# -*- coding: utf-8 -*-
"""
Assignment 
1.( Two-Sample/Independent T-Test)
    Use the dataset women_shoe_prices.csv
    
    Use the 'Colors column for analysis
    
    Do pink-color shoes cost more than other color shoes?
    (Note: look for pink as the only or unique color, ignore mixed/multiple colors for comparison)

2.In the cricket example, do SR Tendulkar and V Kohli have a similar batting performance?

3.In the cricket example, try to find if there is a significant difference between the bowling average of
    1) Teams: England, Australia, India, Pakistan
    2) Players: SK Warne, A Kumble, JA Snow, Wasim Akram
    (Note: Bowling average = Runs conceded divided by Wickets taken)
    Use datasets Men Test Player Innings Stats 19th, 20th and 21st century.




"""

import pandas as pd
import numpy as np

df = pd.read_csv(r"F:\data_analytics\dataset\women_shoe_prices.csv")
df.columns


pink_cost = df[df['colors']=='Pink']['prices.amountMax']
other_cost = df[df['colors']!='Pink']['prices.amountMax']


#Perform t-test
import scipy.stats as stats
import matplotlib.pyplot as plt
t_statistic, p_value = stats.ttest_ind(pink_cost,other_cost)
t_statistic = abs(t_statistic) #Making the t_statistic value positive

#Print results
print()
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
print()
print("Critical Value Conclusion:")
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
    


#%%
'''
3.In the cricket example, try to find if there is a significant difference between the bowling average of
    1) Teams: England, Australia, India, Pakistan
    2) Players: SK Warne, A Kumble,JA Snow, Wasim Akram
    (Note: Bowling average = Runs conceded divided by Wickets taken)
    Use datasets Men Test Player Innings Stats 19th, 20th and 21st century.
'''
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

cent_19 = pd.read_csv(r"F:\data_analytics\dataset\Men Test Player Innings Stats - 19th Century.csv")
cent_20 = pd.read_csv(r"F:\data_analytics\dataset\Men Test Player Innings Stats - 20th Century.csv")
cent_21 = pd.read_csv(r"F:\data_analytics\dataset\Men Test Player Innings Stats - 21st Century.csv")

cent_19=cent_19.drop_duplicates()
cent_20=cent_20.drop_duplicates()
cent_21=cent_21.drop_duplicates()

df = pd.concat([cent_19,cent_20,cent_21])
#df.columns
#df.info()

#1) Teams: England, Australia, India, Pakistan
England = df[df['Country']=='England']
Australia = df[df['Country']=='Australia']
India = df[df['Country']=='India']
Pakistan = df[df['Country']=='Pakistan']

#Innings Runs Conceded data cleaning
England = England.dropna(subset=['Innings Runs Conceded'])
England['Innings Runs Conceded']=England['Innings Runs Conceded'].replace('-',0).astype('int')

Australia = Australia.dropna(subset=['Innings Runs Conceded'])
Australia['Innings Runs Conceded']=Australia['Innings Runs Conceded'].replace('-',0).astype('int')

India = India.dropna(subset=['Innings Runs Conceded'])
India['Innings Runs Conceded']=India['Innings Runs Conceded'].replace('-',0).astype('int')

Pakistan = Pakistan.dropna(subset=['Innings Runs Conceded'])
Pakistan['Innings Runs Conceded']=Pakistan['Innings Runs Conceded'].replace('-',0).astype('int')

#Innings Wickets Taken data cleaning
#England['Innings Wickets Taken'].unique()
England['Innings Wickets Taken']=England['Innings Wickets Taken'].replace('-',0).astype('int')
Australia['Innings Wickets Taken']=Australia['Innings Wickets Taken'].replace('-',0).astype('int')
India['Innings Wickets Taken']=India['Innings Wickets Taken'].replace('-',0).astype('int')
Pakistan['Innings Wickets Taken']=Pakistan['Innings Wickets Taken'].replace('-',0).astype('int')


#Bowling Average
#England.columns
England['Bowling Average'] = England['Innings Runs Conceded']/England['Innings Wickets Taken']
Australia['Bowling Average'] = Australia['Innings Runs Conceded']/Australia['Innings Wickets Taken']
India['Bowling Average'] = India['Innings Runs Conceded']/India['Innings Wickets Taken']
Pakistan['Bowling Average'] = Pakistan['Innings Runs Conceded']/Pakistan['Innings Wickets Taken']

#Dropping nan and infinity values
England['Bowling Average'] = England['Bowling Average'].replace([np.inf,-np.inf],np.nan)
England = England.dropna(subset=['Bowling Average'])

Australia['Bowling Average'] = Australia['Bowling Average'].replace([np.inf,-np.inf],np.nan)
Australia = Australia.dropna(subset=['Bowling Average'])

India['Bowling Average'] = India['Bowling Average'].replace([np.inf,-np.inf],np.nan)
India = India.dropna(subset=['Bowling Average'])

Pakistan['Bowling Average'] = Pakistan['Bowling Average'].replace([np.inf,-np.inf],np.nan)
Pakistan = Pakistan.dropna(subset=['Bowling Average'])


#England['Bowling Average'].unique()
#Pakistan['Bowling Average'].unique()

#One Way Anova
import numpy as np
from scipy.stats import f_oneway

method_A = England['Bowling Average']
method_B = Australia['Bowling Average']
method_C = India['Bowling Average']
method_D = Pakistan['Bowling Average']


#Perform ONE WAY ANOVA
f_statistic, p_value = f_oneway(method_A, method_B, method_C,method_D)
print()
print("F statistic: ",f_statistic)
print("P-value: ",p_value)

alpha =0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the bowling average of Teams: England, Australia, India, Pakistan")
else:
    print("Failed to reject the null hypothesis: No there is a significant difference between the Teams: England, Australia, India, Pakistan")


#2) Players: SK Warne, A Kumble, JA Snow, Wasim Akram
England.columns
England['Innings Player'].unique()
sk_warne = Australia[Australia['Innings Player'] == 'SK Warne']
#print(sk_warne['Country'])
a_kumble = India[India['Innings Player'] == 'A Kumble']
#print(a_kumble['Country'])
ja_snow = England[England['Innings Player'] == 'JA Snow']
#print(ja_snow['Country'])
wasim_akram = Pakistan[Pakistan['Innings Player'] == 'Wasim Akram']
#print(wasim_akram['Country'])

sk_warne.columns


#One Way Anova
import numpy as np
from scipy.stats import f_oneway

method_A = sk_warne['Bowling Average']
method_B = a_kumble['Bowling Average']
method_C = ja_snow['Bowling Average']
method_D = wasim_akram['Bowling Average']


#Perform ONE WAY ANOVA
f_statistic, p_value = f_oneway(method_A, method_B, method_C,method_D)
print()

print("F statistic: ",f_statistic)
print("P-value: ",p_value)

alpha =0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the bowling average of Players: SK Warne, A Kumble, JA Snow, Wasim Akram")
else:
    print("Failed to reject the null hypothesis: No there is a significant difference between the Players: SK Warne, A Kumble, JA Snow, Wasim Akram")
































