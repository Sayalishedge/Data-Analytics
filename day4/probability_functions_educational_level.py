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

#PMF provides a bar method that plots the values and their probabilities as a bar chart.
pmf_educ_norm.bar(label='EDUC')

plt.xlabel('Years of education')
plt.xticks(range(0, 21, 4))
plt.ylabel('PMF')
plt.title('Distribution of years of education')
plt.legend();
plt.show()



'''
CDF
'''
from empiricaldist import Cdf

#Age 98 and 99 mean do not know and do not answer -so replace them
age = gss['AGE'].replace([98, 99],np.nan)

cdf_age = Cdf.from_seq(age)
cdf_age.plot()

plt.xlabel('Age (years)')
plt.ylabel('CDF')
plt.title('Distribution of age');
plt.show()
#The x-axis is the ages, from 18 to 89. The y-axis is the cummulative probabilities from 0 to 1.


#We can also obtain the cumulative prob up to a certain point. eg. age 51
q = 51
p = cdf_age(q)
print(p)
#about 63% of the respondents are 51 years old or younger

#Inversly, find the age ata a certain value of cummulative probability;
p1 = 0.25
q1 = cdf_age.inverse(p1)
print(q1)
# 25% of the respondents are age 31 or less. Another way to say the same thing is
# age 31 is the  25th percentile of this distribution.

#We can also use 75th percentile to find IQR
#It measures the spread of the distribution, so it is similar to standard deviation or variance.

p3 = 0.75
q3 = cdf_age.inverse(p3)
print(p3)
print(q3-q1)
#Plot the CDF
plt.plot(sorted_age,cdf)


#Now lets compare PMF and CMF
#Create series for male and frmale respondents
male = (gss['SEX'] == 1)
female = (gss['SEX'] == 2)

#Select ages
male_age = age[male]
female_age = age[female]

#Plot PMF for each
pmf_male_age = Pmf.from_seq(male_age)
pmf_male_age.plot(label='Male')

pmf_female_age = Pmf.from_seq(female_age)
pmf_female_age.plot(label='Female')

plt.xlabel('Age (years)')
plt.ylabel('PMF')
plt.title('Distribution of age by sex')
plt.legend();
plt.show()

#Now CDF for same data_file
cdf_male_age = Cdf.from_seq(male_age)
cdf_male_age.plot(label='Male')

cdf_female_age = Cdf.from_seq(female_age)
cdf_female_age.plot(label='Female')

plt.xlabel('Age (years)')
plt.ylabel('PMF')
plt.title('Distribution of age by sex')
plt.legend();
plt.show()

'''
Observations:
    In general, CDFs are smoother than PMFs 
'''

#C
cdf_age = empiricaldist.Cdf.from_seq(age)

cdf_values = cdf_age.values
data_points = cdf_age.index
plt.plot(data_points, cdf_values)

plt.axvline(x=q1,color='red',linestyle='--', label='Q1')
plt.axvline(x=q3,color='orange', linestyle='--',label='Q3')

plt.xlabel('Age (years)')
plt.ylabel('CDF')
plt.title('Distribution of agewith Q1 and Q3 marked')
plt.legend();
plt.show()


#Annotate Q1 and Q3 on the plot
#Calculate quartiles (Q1 and Q3)


'''
PDF
Create a histogram to visuaize the PDF/Kernel Density Plot
'''

age_data=gss['AGE']
plt.hist(age_data, bins=20, density=True, alpha=0.6, color='b', label='PDF')
plt.xlabel('Age')
plt.ylabel('Probability Density')
plt.title('PDF of Age in GSS Dataset')
plt.legend()
plt.show()

#Corresponding PMF
pmf = age_data.value_counts(normalize=True).sort_index()
#Create a bar plot of the PMF
plt.figure(figsize=(10, 6))
plt.bar(pmf.index, pmf.values)
plt.xlabel('Age')
plt.ylabel('Probability')
plt.title('PMF of Age in GSS Dataset')
plt.xticks(rotation=90)
plt.show()
























