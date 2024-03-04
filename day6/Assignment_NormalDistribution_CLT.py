# -*- coding: utf-8 -*-
"""
Assignment
For a particular player (eg. Sachin Tendulkar), check if his scores follow normal distribution.
(Hint: Use shapiro test, Draw a Q-Q plot)

Use CLT to see if it is true for this data. 
Draw vizualizations if necessary

Compare Sachin's batting average with the Indian team's total batting average. Use a Z-test to access
the statistical significance of the difference. Display the Z-score and p-value
(Hint: Use the ztest() function)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

cent_20 = pd.read_csv(r"F:\data_analytics\dataset\Men ODI Player Innings Stats - 20th Century.csv")
cent_21 = pd.read_csv(r"F:\data_analytics\dataset\Men ODI Player Innings Stats - 21st Century.csv")
cent_21=cent_21.drop_duplicates() #cent_21 had repeated rows thrice

df = pd.concat([cent_20,cent_21])
print(df.columns)

#selecting data for Sachin Tendulkar
df_sachin = df[df['Innings Player'] == 'SR Tendulkar']

df_no_duplicates = df_sachin.drop_duplicates(subset=['Innings Player','Innings Date'])
df_sachin = df_no_duplicates.copy()

df_sachin['Innings Runs Scored Num'] = pd.to_numeric(df_sachin['Innings Runs Scored Num'],errors='coerce')
df_sachin = df_sachin.dropna(subset=['Innings Runs Scored Num']).astype({'Innings Runs Scored Num' : 'int'})

print(df_sachin.columns)
score = df_sachin['Innings Runs Scored Num']
plt.figure(figsize=(8,6))
plt.hist(score, bins=5, edgecolor='black')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title("Distribution of sachin's score")




#Normalize test with Shapiro-Wilk test
shapiro_test = stats.shapiro(score)
print("Shapiro-Wilk p-value:", shapiro_test.pvalue)

# Q-Q plot
stats.probplot(score, fit=stats.norm, plot=plt)
plt.xlabel("")
plt.ylabel("")
plt.title("Q-Q Plot for Sachin's scores")
plt.grid(True)
plt.show()


#Interpretation
if shapiro_test.pvalue < 0.05:
    print("The data likely does not follow a normal distribution.")
else:
    print("The data may be normally distributed, but the Q-Q plot can provide further insight.")
  
 

#CLT  
plt.figure(figsize=(21,25))

#Check variious column distributions
plt.subplot(2,3,1)
score.plot(kind='hist')
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Score Distribution')
plt.show()  
  
means_list = [score.sample(50,replace = False).mean() for i in range(452)]
print(means_list)    
  
#Plot histogram of means_list
plt.hist(means_list, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Mean Score')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Means (Scores)')
plt.show()




#Z-test

from statsmodels.stats .weightstats import ztest
sachin_avg = round(score.mean() )

df_india = df[df['Country']=='India']
df_india['Innings Runs Scored Num']    
  
df_no_duplicates = df_india.drop_duplicates(subset=['Innings Player','Innings Date'])
df_india = df_no_duplicates.copy()

df_india['Innings Runs Scored Num'] = pd.to_numeric(df_india['Innings Runs Scored Num'],errors='coerce')
df_india = df_india.dropna(subset=['Innings Runs Scored Num']).astype({'Innings Runs Scored Num' : 'int'})
print(df_india.columns)  
  
india_avg = round(df_india['Innings Runs Scored Num'].mean()  )
    
z_statistic,p_value = ztest(score, value=india_avg)
print(f"Z=Statistic: {z_statistic}, p-value: {p_value}")  
  
#Now vizualize the original data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Fit a normal distribution to the data
mu, std = norm.fit(score)

#Plot the histogram
plt.hist(score,bins=10, density=True, alpha=0.6, color='g')

#Plot the PDF of the fitted normal distribution
xmin,xmax =plt.xlim()
x = np.linspace(xmin,xmax,100)
p = norm.pdf(x,mu,std)
plt.plot(x,p,'k',linewidth=2)
plt.xlabel('')
plt.ylabel('')
plt.title('Fit results: mu =%.2f, std =%.2f' % (mu,std))
plt.show()

#Show hypothesized mean
ecdf = np.arange(1, len(score) + 1) / len(score)

percentile_at_test_value = np.interp(india_avg,np.sort(score),ecdf)

plt.step(np.sort(score),ecdf, label='Empirical CDF', where='post')
plt.axvline(india_avg, color='red', linestyle='dashed',linewidth=2, label='Test Value')
plt.xlabel('')
plt.ylabel('')
plt.title('Empirical Cumulative Distribution Function(ECDF)')
plt.legend()
plt.show()  
    
#Plotting the data
player ='SR Tendulkar'
plt.figure(figsize=(10,6))  
sns.boxplot(x='Innings Player', y='Bowling Average', data=score, color='skyblue')
plt.axhline(india_avg, color='red', linestyle='dashed', linewidth=2, label='Overall Average')
plt.title(f"Bowling Average for {player} vs. Overall Average")
plt.xlabel('Player')
plt.ylabel('Bowling Average')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
    
  
    