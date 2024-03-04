# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:03:43 2023

@author: dbda
"""

import numpy as np
import pandas as pd

#pip install yfinance==0.2.28

import yfinance as yf
tickers = ['TCS.NS',"ICICIBANK.NS", "RELIANCE.NS", "BHARTIARTL.NS", "ITC.NS", "MARUTI.NS", "BAJFINANCE.NS"]

df = yf.download(tickers,period="3y")
df.swaplevel(axis=1)
close=df.Close

#Make base price 100 to bring all on a common scale
norm=close.div(close.iloc[0]).mul(100)
print("Printing normal")
print(norm)

#The code calculates the daily percentage returns for each stock and stores them in the ret Dataframe.
ret=close.pct_change().dropna()
print(ret)
print(ret.cov())

#The code calculates the daily percentage returns for each stock and stores them in the ret Dataframe
ret=close.pct_change().dropna()
print(ret)
print(ret.cov())
print(ret.corr())

import matplotlib.pyplot as plt
import seaborn as sns

#First correlation
plt.figure(figsize=(15,7))
sns.heatmap(ret.corr(),cmap="Blues")
plt.show()

#Now correlation with annotations
plt.figure(figsize=(15,7))
sns.heatmap(ret.corr(),cmap="Blues", annot=True)
plt.plot()
plt.show()

#Add Covariance
plt.figure(figsize=(15,7))
sns.heatmap(ret.cov(),cmap="Blues", annot=True)
plt.show()

















