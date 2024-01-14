# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:23:42 2023

@author: dbda
"""
import numpy as np
import pandas as pd
df = pd.read_csv(r"F:\data_analytics\day1\Dummy_Sales_Data_v1.csv")
#sorting
df_sorted = df.sort_values(by=["Quantity"],ascending=True)
df_sorted

#counting
print(df.value_counts("Sales_Manager"))

#min, max
print(df.nlargest(10,"Delivery_Time(Days)"))
print(df.nsmallest(7,"Shipping_Cost(USD)"))

df1=df.copy()
#rename
df1.rename(columns = {"Shipping_Cost(USD)":"Shipping_Cost",
                      "Delivery_Time(Days)":"Delivery_Time_In_Days"},inplace=True)

print(df.info())
print(df1.info())

condition = df1['Status'] == "Not Shipped"
print(condition)

result=df.where(condition)
print(result)

#remove a column
df1.drop("OrderCode",axis=1,inplace=True)
print(df1.info())

#drop a row/column
df1.head()
df1.drop(3,axis=0,inplace=True)
df1.head()

df1.drop(3,axis=1,inplace=True)
df1.info()

#change index
print(df.head())
df.set_index('Sales_Manager',inplace=True)
print(df.head())


































