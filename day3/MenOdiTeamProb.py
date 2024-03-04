# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:53:51 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df20 = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\Men ODI Team Match Results - 20th Century.csv")
df21 = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\Men ODI Team Match Results - 21st Century.csv")

print(df20)
print(df21)
print(df20.columns)
print(df21.columns)

df = pd.concat((df20,df21))
print(df.columns)

print(df['Country'].unique())

print(df['Result'].unique())


#created a copy
df1=df.copy()
print(df1.columns)
print(df1['Result'].unique())
df1.drop(df1[df1['Result'] == 'Aban'].index, inplace = True)
df1.drop(df1[df1['Result'] == 'Canc'].index, inplace = True)
df1.drop(df1[df1['Result'] == 'Tied'].index, inplace = True)
df1.drop(df1[df1['Result'] == 'N/R'].index, inplace = True)
print(df1['Country'].unique())

print(df1['Result'].unique())

print(df1['Home/Away'].unique())

#Select a country India
df_india = df1[df1['Country']=='India']
print(df_india['Country'].unique())
print(df_india.columns)

print(df_india['Home/Away'])
print(df_india['Result'])



#A=Playing at home
#B = winning
#use bayes theorem to find P(A|B) P(B|A)


p_a = len(df_india[df_india['Home/Away']=='Home'])/len(df_india)
#print(p_a)

p_b = len(df_india[df_india['Result']=='Won'])/len(df_india)
#print(p_b)

p_a_and_b = len(df_india[(df_india['Home/Away']=='Home') & (df_india['Result']=='Won')])/len(df_india)
#print(p_a_and_b)

#P(A|B) probability of playing on home ground if won
p_a_given_b = round(p_a_and_b/p_b,2)
print("Probability of India on home ground if won =",p_a_given_b)

#P(B|A) Probability of winning given home ground is India
p_b_given_a = round((p_a_given_b*p_b)/p_a,2)
print("Probability of winning given home ground is India =",p_b_given_a)


#A=Playing Away
#B=Winning
p_a1 = len(df_india[df_india['Home/Away']=='Away'])/len(df_india)
#print(p_a1)

p_b1 = len(df_india[df_india['Result']=='Won'])/len(df_india)
#print(p_b1)

p_a1_and_b1 = len(df_india[(df_india['Home/Away']=='Away') & (df_india['Result']=='Won')])/len(df_india)
#print(p_a1_and_b1)

#P(A|B) probability of playing not on home ground if india wins
p_a1_given_b1 = round(p_a1_and_b1/p_b1,2)
print("Probability of playing not on home ground if india wins =",p_a1_given_b1)

#P(B|A) Probability of winning given home ground is not India
p_b1_given_a1 = round((p_a1_given_b1*p_b1)/p_a1,2)
print("Probability of winning given home ground is not India = ",p_b1_given_a1)


































