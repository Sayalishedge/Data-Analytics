# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:05:44 2023

@author: dbda
"""

import empiricaldist
from statadict import parse_stata_dict

dict_file = 'F:\data_analytics\dataset\GSS.dct'
data_file = 'F:\data_analytics\dataset\GSS.dat.gz'

from statadict import parse_stata_dict
stata_dict = parse_stata_dict(dict_file)
#print(stata_dict)

import gzip
fp = gzip.open(data_file)

#Convert the file into a Pandas Dataframe
import pandas as pd
gss = pd.read_fwf(fp, names=stata_dict.names,colspecs=stata_dict.colspecs)
print(gss.shape)
print(gss.head())

#Distribution of education
print(gss['EDUC'].value_counts().sort_index())

#The values 98 and 99 are special codes for "Dont know" and "No answer". We will use replace to replace these codes with NaN.

import numpy as np
educ = gss['EDUC'].replace([98, 99], np.nan)

#Visualize it
import matplotlib.pyplot as plt
educ.hist(grid=False)
plt.xlabel('Years of education')
plt.ylabel('Number of respondents')
plt.title('Histogram of education level')
plt.show()

'''
Looks like the peak is near 12 years of education. But a histogram is not the best way 
to visualize this distribution because it obscures some important details.
An alternative is to use a PMF
'''
from empiricaldist import Pmf
pmf_educ = Pmf.from_seq(educ, normalize=False)
print(type(pmf_educ))
print(pmf_educ.head())

#In this dataset, there are 165 respondents who report that they have no formal education, and 47 who have only one year. Here the last few rows.
print(pmf_educ.tail())

#There are 1439 respondents who report that they have 20 or more years of formal education, which probably means they attended college and graduate school.
#Get the count for 20 years of education separately
print(pmf_educ[20])

'''
Usually when we make a PMF, we want to know the fraction of respondents with each value,
rather than the counts. We can do that by setting normalize=True, then we get a normalized
PMF, that is, a PMF where the values in the second column add up to 1.
'''
pmf_educ_norm = Pmf.from_seq(educ, normalize=True)
print(pmf_educ_norm.head())
print(pmf_educ_norm[12]) # Sample for 12 years of experience




























