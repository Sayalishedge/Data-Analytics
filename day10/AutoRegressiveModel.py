# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:46:40 2023

@author: dbda
"""

import pandas as pd
import seaborn as sns

df = sns.load_dataset('flights')
print(df.head)
#We get only ear and month for the date
                 