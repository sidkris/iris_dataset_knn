#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 00:27:54 2020

@author: sid
"""

# working on the iris dataset using knn (n=1 in this example)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

iris_dataset = load_iris()
#print("Keys of iris dataset:\n {}".format(iris_dataset.keys())) 
#print("target names: {}".format(iris_dataset["target_names"]))
#print("feature names: {}".format(iris_dataset["feature_names"]))
#print(iris_dataset["data"].shape)
#print(iris_dataset["target"].shape)

#the "train_test_split" function in scikit learn will shuffle and split the available data to training (75%) Vs 
#test (25%)

X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

print("We will be using {} training cases".format(X_train.shape[0]))
print("And, we will be using {} test cases".format(X_test.shape[0]))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
#print(iris_dataframe)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o') #hist_kwds={'bins'=20}, s=60, alpha=0.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#--testing our algorithm--

y_pred = knn.predict(X_test)
print("Test set predictions : \n {}". format(y_pred))
      
# evaluating the algorithm's performance
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#alternative method to evaluate performance
#print("Test set score: {:.2f}".format(knn.score(X_test,y_test)))


      
      
      