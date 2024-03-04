# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:54:19 2023

@author: dbda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset('flights')
print(df.head())
#convert the date to yyyy-mm-dd format in a new column named yearMonth
df['yearMonth']="01-"+df['month'].astype(str)+"-"+df['year'].astype(str)

#yearMonth is of type object we may have problems later so convert it inot datetime

df['yearMonth']=pd.to_datetime('01-'+df['month'].astype(str)+'-'+df['year'].astype(str))

print(df.info)
print(df.head())

#make yearMonth columns as the DF index

df.set_index('yearMonth', inplace=True) # inplace will make the change permanent to the df
print(df.head())

#now plot
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x=df.index, y=df.passengers)
plt.show()

#the graph will show patterns(eg.seasonality - data going up and down)
#refer to the slides for explanations
#we see in our graph two pattern :seasonality and trend

#Rolling Statistics
#calculate and plot rolling mean and standard deviation for 12 months
df['rollMean']=df.passengers.rolling(window=12).mean()
df['rollStd']=df.passengers.rolling(window=12).std()

print(df['rollMean'])
print(df['rollStd'])

plt.figure(figsize=(10,5))
sns.lineplot(data=df,x=df.index,y=df.passengers)
sns.lineplot(data=df,x=df.index,y=df.rollMean)
sns.lineplot(data=df,x=df.index,y=df.rollStd)
plt.show()
#conclusion:Mean is not stationary ,SD is stationary ; so our data is not stationary

#ADF(Augmented Dicky-Fuller Test)
from statsmodels.tsa.stattools import adfuller

adfTest=adfuller(df['passengers'])
print(adfTest)
#let us interpret these values below by converting into a series
stats=pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used', 'number of observations used'])
print(stats)

for key,values in adfTest[4].items():
    print('criticality',key,":",values)

#we will see that our test statistic > critical value in all the cases ,
#so we do not reject the null hypothesis. It means that out data is not stationary



def test_stationary(dataframe, var):
    dataframe['rollMean']=dataframe[var].rolling(window=12).mean()
    dataframe['rollStd']=dataframe[var].rolling(window=12).std()
    from statsmodels.tsa.stattools import adfuller
    
    adfTest=adfuller(dataframe[var])
    stats=pd.Series(adfTest[0:4],index=['Test Statistic','p-value','#lags used', 'number of observations used'])
    print(stats)
    for key,values in adfTest[4].items():
        print('criticality',key,":",values)
    plt.figure(figsize=(10,5))
    sns.lineplot(data=dataframe,x=dataframe.index,y=var)
    sns.lineplot(data=dataframe,x=dataframe.index,y='rollMean')
    sns.lineplot(data=dataframe,x=dataframe.index,y='rollStd')
    plt.show()    

#By default, shift is by 1 time period (here, one month)
#Create a new column which will contain the shifted value from passengers column
air_df=df[['passengers']].copy()
air_df['shift'] = air_df.passengers.shift(10)
air_df['shiftDiff'] = air_df['passengers'] - air_df['shift']
print(air_df.head(20))

#Test stationary
test_stationary(air_df.dropna(),'shiftDiff')

#Conclusion:
    
    
#ARIMA  
#Create coulumns for one month and one year lagged data
airP = df[['passengers']].copy(deep=True)
airP['firstDiff'] = airP['passengers'].diff()
airP['Diff12'] = airP['passengers'].diff(12) #This will be used later in SARIMAX
  
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_pacf(airP['firstDiff'].dropna(),lags=20)    
plt.show()

#Shaded area is insignificant area
#PACF gives us the auto regressive values (i.e p-Refer to the slides)
#First 'p' is 1 (the x-axis coordinate), whose value is ~0.31 (the y-axis coordinate)
#So significant p values are 1, 2, 4, 6, etc
#Now let us take this value as p and find q, for which we need ACF 
plot_acf(airP['firstDiff'].dropna(),lags=20)
plt.show()

#Results of ACF are similar to that of PACF
#Interpretation: We got q. Significant q values are 1, 3, 4, 8, etc)
#Let us take p=1, q=3 (both are significant) and d=1 (already known)


#Build ARIMA model
train = airP[:round(len(airP)*70/100)] #Take the first 70% data
print(train.tail()) #Just to check where it ends
test = airP[round(len(airP)*70/100):] #Take the last 30% data, starting from 71%
print(test.head()) # Just to check where it starts

model = ARIMA(train['passengers'],order=(1,1,3)) #Parameters: p,d,q
model_fit = model.fit()
prediction = model_fit.predict(start=test.index[0], end=test.index[-1])
airP['arimaPred'] = prediction
print(airP.tail())

#Plot
sns.lineplot(data=airP,x=airP.index,y='passengers')
sns.lineplot(data=airP,x=airP.index,y='arimaPred')
plt.show()

#Conclusion: The ARIMA prediction is not good.


#Build SARIMAX predicitons
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['passengers'],order=(1,1,3),seasonal_order=(2,1,2,12))
model_fit = model.fit()
prediction = model_fit.predict(start=test.index[0], end=test.index[-1])
airP['sarimaxPred'] = prediction
print(airP.tail())

#Plot
airP.dropna()
print(airP.head())
sns.lineplot(data=airP,x=airP.index,y='passengers')
sns.lineplot(data=airP,x=airP.index,y='sarimaxPred')
sns.lineplot(data=airP,x=airP.index,y='arimaPred')
plt.show()

#Conclusion: Compared to ARIMA, SARIMAX is much better
#Future Prediction
futureDate = pd.DataFrame(pd.date_range(start='1961-01-01', end='1962-12-01', freq='MS'),columns=['Dates'])
futureDate.set_index('Dates', inplace=True)
print(futureDate.head())

#Predict and print
print(model_fit.predict(start=futureDate.index[0], end=futureDate.index[-1]))

#Plot
airP.dropna()
sns.lineplot(data=airP, x=airP.index, y='passengers')
sns.lineplot(data=airP, x=airP.index, y='sarimaxPred')
sns.lineplot(data=airP, x=airP.index, y='arimaPred')
model_fit.predict(start=futureDate.index[0], end=futureDate.index[-1]).plot(color='black')
plt.show()
















