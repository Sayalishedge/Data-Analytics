# -*- coding: utf-8 -*-
"""
Assignment:
    Do multivariate analysis using any three columns of the tips dataset with the tips column.
    Draw graphs/plots, as appropriate. 
    What is your test inference?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv(r"F:\data_analytics\dataset\tips.csv")
dataset.columns
dataset.day.unique()

dataset['sex'] = dataset['sex'].map({'Female':1,'Male':0})
dataset['smoker'] = dataset['smoker'].map({'Yes':1,'No':0})
print(dataset)

x = dataset[['total_bill', 'sex', 'smoker']]
y = dataset['tip']

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

from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootmeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print("R Squared : {:.2f}".format(mlr.score(x,y)*100))
print("Mean Absolute Error : ",meanAbErr)
print("Mean Square Error: ",meanSqErr)
print("Root Mean Square Error: ",rootmeanSqErr)

print(y_test, y_pred_mlr)

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
Intercept :  1.1427192734805904
Coefficient :  [ 0.09173219 -0.10275381 -0.00190105]

R Squared : 44.51
Mean Absolute Error :  0.7504401360529248
Mean Square Error:  1.072010083950527
Root Mean Square Error:  1.0353791981445866

Output Interpretation:
    
   
    
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv(r"F:\data_analytics\dataset\tips.csv")
dataset.columns
dataset.day.unique()

print(dataset)

col = dataset[['total_bill','tip']]



#Correlation Analysis
correlation_matrix  = col.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

#Pairplot --Vizualize scatterplots for multiple numerical variables to explore pairwise relationship
sns.pairplot(dataset[['total_bill','tip','size']])
plt.show()

#Categorical Analysis -- Categorical variables relate to numeric variables
sns.boxplot(x='sex',y='total_bill', data=dataset)
plt.show()

#Regression Analysis
sns.regplot(x='total_bill', y='tip', data=dataset)
plt.show()


