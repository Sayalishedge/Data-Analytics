# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:08:50 2023

@author: dbda
"""

import numpy as np
import pandas as pd
data={'Name' : ['A','B','C','D','E','F','G','H','I','J','K','L','M','N'],
      'Values' : [4,7,np.nan,80,76,np.nan,np.nan,34,12,np.nan,8,45,100,np.nan]}

df = pd.DataFrame(data)
print(df)
print(df.info())



#mean imputation
df_mean_imputed = df.copy()
mean = df['Values'].mean()
print(mean )
df_mean_imputed = df['Values'].fillna(mean)
print("After mean imputaiton \n",df_mean_imputed)

#median imputation
df_median_imputed = df.copy()
med = df['Values'].median()
print(df)
df_median_imputed_imputed = df['Values'].fillna(med)
print("After median imputation \n",df_median_imputed_imputed)

#forward fill
print(df)
df_ffill = df.copy()
df_ffill = df_ffill.fillna(method='ffill')
print("After forward fill \n",df_ffill)

#backward fill
print(df)
df_bfill = df.copy()
df_bfill = df_bfill.fillna(method='bfill')
print("After backward fill \n",df_bfill)

#linear interpolation
print(df)
df_linear_interpolation = df.copy()
df_linear_interpolation = df_linear_interpolation.interpolate()
print("After linear interpolation \n",df_linear_interpolation)

