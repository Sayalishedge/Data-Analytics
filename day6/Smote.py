# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:18:59 2023

@author: dbda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv(r"F:\data_analytics\dataset\creditcard.csv")
print(data.head(3))

#V1 to V28 are encoded columns, original contents suppressed because of confidentiality
#Plot the histogram of class variable
pd.value_counts(data['Class']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
#Plot shows data is imbalanced
data['Class'].value_counts()

#Standardize the 'Amount' coulumn and drop the 'Time' and 'Amount' columns 
from sklearn.preprocessing import StandardScaler 
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

#Now that the amount has been scaled to a normal distribution of  -1 to 1, remove the original amount column and also the
#time column.
data = data.drop(['Time','Amount'],axis=1)
print(data.head())

#Separate featues (X) and target variable (y)
X = np.array(data.iloc[:,data.columns != 'Class'])
y = np.array(data.iloc[:,data.columns == 'Class'])
print('Shape of X : {} '.format(X.shape))
print(f"Shape of y : {y.shape}")

#Split the dataset into training and testing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)

#Print info about the shapes of the training and testing sets
print(f"Number transactions in X-train dataset = {X_train.shape}")
print(f"Number transactions in X-test dataset = {X_test.shape}")
print(f"Number transactions in y-train dataset = {y_train.shape}")
print(f"Number transactions in y-test dataset = {y_test.shape}")

#Print class distribution before and after oversampling with SMOTE
print("Before OverSampling, counts of label '1' : {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0' : {} \n".format(sum(y_train==0)))


sm = SMOTE(random_state=2)#put a fixed seed for reproducability

X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
print('After OverSampling, the shape of train_X : {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y : {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1' : {} ".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0' : {} ".format(sum(y_train_res==0)))

#Use GridSearchCSV to find the best hyperparameter for Logistic Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#defines a dictionary parameters where the key is 'C', representing the regularization
#parameter in logistic regression, and the associated values are generated using np.linspace(1,10,10).
parameters = {'C' : np.linspace(1,10,10)}
lr = LogisticRegression()

'''
lr: The base estimatior(Logistic Regression in this case).

parameters:The dictionary of hyperparameter values to search over.

cv=5: The number of folds for cross-validation. It performs 5-fold cross-validation,dividing the dataset
      into 5 subsets and training the model 5 times, each time using a different subset as the test set.

verbose=5 : Controls the verbosity during the grid search. A higher value (5 in this case) means more information
            will be printed during the search.

n_jobs=3: Number of CPU cores to use for parallel computation. Setting it to 3 means using 3 CPU cores for faster computation.
'''

clf = GridSearchCV(lr, parameters,cv=5,verbose=5,n_jobs=3)

'''
The fit method is called on the GridSearchCV object, and it performs an exhaustive search over the
specified parameter values. It finds the best hyperparameters that maximize the performance metric (default is accuracy)
based on cross-validation. The training data (X_train_res, overssampled using SMOTE, and y_train_res)
is used for this process.
The ravel() function in NumPy is used to flatten or reshape arrays. When applied to an array, it returns
a contiguous flattened array, i.e., a one-dimentional array containing all the elements of the original array.
'''

clf.fit(X_train_res, y_train_res.ravel())
print(clf.best_params_)

#Train Logistic Regression with the best hyperparameter
lr1 = LogisticRegression(C=4,penalty='l1',solver='liblinear',verbose=5)
lr1.fit(X_train_res, y_train_res.ravel())

#Funciton to plot confusion matrix
import itertools
from sklearn.metrics import confusion_matrix,roc_curve,auc

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment = 'center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()    
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')

    
#Evaluate the model on the training set
y_train_pre = lr1.predict(X_train)    
cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

#Print recall metric on the training dataset
print("Recall metric in the train dataset: {}%"
      .format(100 * cnf_matrix_tra[1,1] / (cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))

#Plot confusion matrix for the training set
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra, classes=class_names, title='Confusion matrix - Training Set')
plt.show()



 
#Evaluate the model on the testing set.   
y_pre = lr1.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pre)

#Print recall metric on the testing dataset
print("Recall metric in the testing dataset: {}%"
      .format(100 * cnf_matrix[1,1] / (cnf_matrix[1,0]+cnf_matrix[1,1])))

#Plot confusion matrix for the testing set
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix - Testing Set')
plt.show()



























    




































