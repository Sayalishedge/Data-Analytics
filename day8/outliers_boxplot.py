# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:57:44 2023

@author: dbda
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

tips=sns.load_dataset("tips")

print(tips) 

ax = sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()

#Same thing using local file
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\tips.csv")

#Create a group boxplot
plt.figure(figsize=(10,6))
sns.boxplot(
    x="smoker",
    y="total_bill",
    showmeans=True, #show means on top of boxplots
    data = df,
    palette = "Blues"
    )
plt.xlabel("Smoker Status")
plt.ylabel("Total Bill ($)")
plt.title("Distribution fo Total Bills by Smoker Status (tips.csv)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()










