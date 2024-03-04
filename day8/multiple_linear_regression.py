# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:58:55 2023

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r"C:\cdac\dataAnalytics\dataset\advertising-sales.csv")

x = dataset[['tv', 'radio', 'newspaper']]
y = dataset['sales']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=100)

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train)

#intercept Coefficient
print("Intercept : ",mlr.intercept_)
print("Coefficient : ",mlr.coef_)
print(list(zip(x,mlr.coef_)))

#Prediction of test set
y_pred_mlr = mlr.predict(x_test)
#predicted values
print("Prediction for test set: {}".format(y_pred_mlr))

#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value' : y_test, 'Predicted value' : y_pred_mlr})
print(mlr_diff)

'''
Model Evaluation:
    1)R Squared:
        It tells how many points fall on the regression line.
        R-Squared=90.11 i.e 90.11% of the data fall on the regression line
        R-Squared = Variance explained by the model/Total variance
    2)Mean Absolute Error
        The difference between the actual value and the predicted values.
        The lower the value, better is the performance.
        =0 means that your model is a perfect predictor.
        The mean abs error obtained for this model is 1.227 which is pretty good as it is close to 0
    3)Mean Squared error
        =Average of the square of the differnce between the original and predicted values of the data.
        The lower the value, the better is the performance of the model.
        The mean sq error obtained for this model is 2.636 which is pretty good.
    4)Root Mean Square Error
'''
from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootmeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print("R Squared : {:.2f}".format(mlr.score(x,y)*100))
print("Mean Absolute Error : ",meanAbErr)
print("Mean Square Error: ",meanSqErr)
print("Root Mean Square Error: ",rootmeanSqErr)

print(y_test[1], y_pred_mlr[1])

#Plotting actual vs predicted values
residuals=y_test-y_pred_mlr
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_mlr, color='blue')
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted values(Multiple Linear Regression)")
plt.grid(True)
plt.show()




plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=20, kde=True, color='green')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residuals Distribution (Multiple Linear Regression)")
plt.grid(True)
plt.show()

'''
R Squared : 89.59
Mean Absolute Error :  1.0638483124072025
Mean Square Error:  1.8506819941636958
Root Mean Square Error:  1.360397733813055
10.4 20.00625301929181

Output Interpretation:
    Value of intercept is 4.3345, which shows that if we keep the money spent on TV, radio, and 
    newspaper for advertisement as 0, the estimated average sales will be 4.3345
    A single rupee increse in the money spent on TV for advertisement increases sales by 0.0538,
    the money spent on Radio for advertisement increases sales by 0.1100, and he money spent on
    Newspaper for advertisement increases sales by 0.0062
    
'''










