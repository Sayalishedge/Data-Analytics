# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:19:11 2023

@author: dbda
"""

#1.Load Data
import pandas as pd
dataset = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\salary.csv")
dataset.columns
#For X take all the columns except salary
X = dataset.iloc[:,:-1].values

#For y take the second column i.e salary
y = dataset.iloc[:,1].values

#2.Split the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#3. Fit Sample Linear Regression to Training Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#4.Make Prediciton
y_pred = regressor.predict(X_test)
print(y_pred)

#5. Vizualize training set results
import matplotlib.pyplot as plt
#plot the actual data points of training set
plt.scatter(X_train, y_train, color='red')
#plot the regression line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel("Years of expericence")
plt.ylabel("Salary")
plt.show()

#6. Vizualize test set results
import matplotlib.pyplot as plt
#plot the actual data points of test set
plt.scatter(X_test, y_test, color='red')
#plot the regression line
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel("Years of expericence")
plt.ylabel("Salary")
plt.show()

#7.Make new prediciton
new_salary_pred = regressor.predict([[15]])
print("The predicted salary of a person with 15 years of experience is ", new_salary_pred)

#8.Intercept and Coefficient
#Intercept: Salary for 0 years of experience
print("Intercept .. fresher salary: ",regressor.intercept_)
print("Coefficient .. Additional salary for each additional years experience: ",regressor.coef_)











