# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:43:12 2023

@author: dbda
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#generate scores for 100 students with the avg score of 69.5 and std deviation of 15.2
scores = np.random.normal(loc = 69.5, scale=15.2, size=100)
print(scores)


#Calculate mean and std D of sxores

mean=np.mean(scores)
sd=np.std(scores)

#Calculate Z score
z_score = (scores - mean) / sd


#create a table with student numbers , scores and Z-scores 
data = np.column_stack((np.arange(1, 101), scores, z_score))

#Stach Horizontally:
string_data = np.column_stack((np.arange(1, 101 ).astype(str), scores.astype(str), z_score.astype(str))) #convert dtat to string for printing
print("Table of student scores and z-scores:")
print(string_data)

#Plot the distribution of student scores using histogram
plt.figure(figsize=(8,6))
plt.hist(scores, bins=20,edgecolor= "black")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.title("Distribution of sudent scores")
plt.legend()
plt.show()


#Plot the distribution of Z scores using histogram
plt.figure(figsize=(8,6))
plt.hist(z_score, bins=20,edgecolor= "black")
plt.xlabel("Z-Scores")
plt.ylabel("Frequency")
plt.title("Distribution of sudent scores")
plt.legend()
plt.show()



#define outliers
outliers_score = [110,0,10,150,5]

#add out lier score to original score
scores = np.append(scores, outliers_score)


#calculate the mean and std D of the score
mean=np.mean(scores)
sd=np.std(scores)


#Calculate the z-score for each student
z_scores = (scores - mean) / sd

#Identify outliers based on the threshold(e.g 3 standard D)
outlier_threshold = 3
outliers = scores[np.abs(z_scores) > outlier_threshold]

#create a table with student numbers , scores and Z-scores
#106 bcoz we have added 5 additional outliers
string_data = np.column_stack((np.arange(1, 106 ).astype(str), scores.astype(str), z_score.astype(str))) #convert dtat to string for printing
print("Table of student scores and z-scores:")
print(string_data)

#Plot the distribution of student scores using histogram
plt.figure(figsize=(8,6))
plt.hist(scores, bins=20,edgecolor= "black")

#highlight outliers with different colors and marker
plt.scatter(z_score[np.abs(z_scores) > outlier_threshold],
            [0] * len(z_scores[np.abs(z_scores) > outlier_threshold]), 
            marker="x", color="red", label="Outliers")

'''
z_score[np.abs(z_scores) > outlier_threshold] =>
select data pts from z scores array that are > outlier_threshold in abs value.(It selects outliers)

[0] * len(z_scores[np.abs(z_scores) > outlier_threshold] => 
create a list of 0s that has same length as he no outliers
sets y coordinate of outliers to 0

'''
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.title("Distribution of sudent scores")
plt.legend()
plt.show()











