# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:35:23 2023

@author: dbda
"""
import pandas as pd
df = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\Dummy_Sales_Data_v1.csv")
print(df.info())
print(df.isnull().sum())

print(df.head())
df['Status'].isnull()
df.sample(5) #random rows
print(df['Delivery_Time(Days)'])
print(df.describe())
print(df.nunique())
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
print(df)
pd.reset_option("display.max_rows",None)
pd.reset_option("display.max_columns",None)
print(df)


#Selecting a subset
print(df.query("Quantity > 95"))

print(df.loc[100,['Sales_Manager','Quantity']])

print(df.iloc[[100,105],[3,6]])

#unique sales managers and its count
print(df['Sales_Manager'].nunique())

print(df['Sales_Manager'].unique())


#create a df wherer product_category is missing
df2=df[df["Product_Category"].isnull()]
print(df2['Product_Category'])


#fill nulls with something
df2.fillna("MissingInfo")
pd.set_option('display.max_columns',None)
print(df2.sample())

df2.info()

df2.fillna("MissingInfo")
pd.set_option('display.max_columns',None)


df3 = df2.fillna("MissingInfo")
pd.set_option('display.max_columns',None)
print(df2)
print(df3)

df2=df2.fillna("Missing values")

pd.reset_option("all")
df

































