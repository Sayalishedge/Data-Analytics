# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:40:05 2023

@author: dbda
"""

#pandas crosstab function
#for analyzing 2 categorical values 

import pandas as pd
import numpy as np
np.random.seed(50)
import matplotlib.pyplot as plt
my_dict={
     'gender': np.random.choice(['male','female'],300),
     'education_level':np.random.choice(['high school','graduate','college'],300),
     'score':np.random.randint(60,100,300)
     }

df=pd.DataFrame(my_dict)
print(df)
#crosstab table for gender and education level
ct=pd.crosstab(df['gender'], df['education_level'])
print(ct)

##crosstab table for gender and education level with mean score
ct_mean=pd.crosstab(df['gender'], df['education_level'],
                    values=df['score'],aggfunc='mean')
print(ct_mean)

#crosstab table for gender and education level with row and column margin
import seaborn as sns
ct_margins=pd.crosstab(df['gender'],df['education_level'],margins=True)
sns.heatmap(ct_margins,cmap='coolwarm',annot=True,fmt='d')
plt.show()
print(ct_margins)


#crosstab table for gender and education level with normalization
ct_norm=pd.crosstab(df['gender'], df['education_level'],normalize=True,margins=True)
print(ct_norm)


#crosstab table for gender and education level with visuaization
ct_viz=pd.crosstab(df['gender'], df['education_level'])
ct_viz.plot(kind='bar',stacked=True)
plt.show()



















