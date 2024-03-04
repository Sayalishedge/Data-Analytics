# -*- coding: utf-8 -*-
"""
Assignment (chi-square/correlation/outlier detection)
Use titanic dataset
1.Draw a correlation plot between PassengerId,Survived, PClass, Age, SibSp, Parch.
2.Do an Outlier detection on Age and create a box plot for the same.
3.Perform a chi-square test among passenger class (column Pclass) and survival (column survived).
Comment on the inference of the result.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1.Draw a correlation plot between PassengerId,Survived, PClass, Age, SibSp, Parch.
df = pd.read_csv(r"F:\data_analytics\dataset\titanic-tested.csv")
df.columns

colnames = ['PassengerId','Survived','Pclass','Age','SibSp','Parch']

col = df.loc[:,colnames]
dataplot = sns.heatmap(col.corr(), cmap="YlGnBu", annot=True)
plt.show()

#2.Do an Outlier detection on Age and create a box plot for the same.
#Create a group boxplot
plt.figure(figsize=(10,6))
sns.boxplot(
    x="Age",
    showmeans=True, #show means on top of boxplots
    data = df,
    palette = "Blues"
    )



#3.Perform a chi-square test among passenger class (column Pclass) and survival (column Survived).
#Create frequency tables (crosstabs) for both datasets using pd.crosstab
#print(df.Pclass)
#print(df.Survived)
from scipy.stats import chi2_contingency


Pclass = pd.crosstab(index=df['Pclass'], columns='count')
Survived = pd.crosstab(index=df['Survived'], columns='count')



# Contingency table.
contingency = pd.crosstab(df['Pclass'],df['Survived'])

# Chi-square test of independence.
c, p_value, dof, expected = chi2_contingency(contingency)

print("p-value : ",p_value)
print("c : ",c)
print("dof: ",dof)
print("expected: ",expected)

#Conclusion
if p_value < 0.05:
    print("Conclusion: Stastically significant difference between Pclass and Survived class (p-value < 0.05) so reject H0")
else:
    print("Conclusion: No Stastically significant difference between Pclass and Survived class (p-value >= 0.05) so do not reject H0")





















