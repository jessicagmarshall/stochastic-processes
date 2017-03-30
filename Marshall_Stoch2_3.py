# -*- coding: utf-8 -*-


#Jessica Marshall
#ECE-302
#Programming Assignment #2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy

#Part 3
#Introduction to pattern classification and machine learning

data = pd.read_excel('/Users/jessicamarshall/Desktop/DataScienceIS/IrisData.xlsx', header = None)
X = data.loc[:, 0:3]
y = data.loc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)   #split data into train and test sets

iris_virginica = X_train.loc[y_train =='Iris-virginica']
iris_versicolor = X_train.loc[y_train == 'Iris-versicolor']
iris_setosa = X_train.loc[y_train == 'Iris-setosa']

mean_virginica = np.mean(iris_virginica.loc[:, 0:3], axis = 0)      #determine mean on training data for each class
mean_versicolor = np.mean(iris_versicolor.loc[:, 0:3], axis = 0)
mean_setosa = np.mean(iris_setosa.loc[:, 0:3], axis = 0)

cov_virginica = np.cov(iris_virginica.loc[:, 0:3], rowvar = False)      #determine covariance on training data for each class
cov_versicolor = np.cov(iris_versicolor.loc[:, 0:3], rowvar = False)
cov_setosa = np.cov(iris_setosa.loc[:, 0:3], rowvar = False)

#X_train_mat = X_train.as_matrix()
X_test_mat = X_test.as_matrix()
#y_train_mat = y_train.as_matrix()
y_test_mat = y_test.as_matrix()

test = np.zeros(y_test_mat.shape[0])

for i in range(0, y_test.shape[0]):
    

#pdf_virginica = scipy.stats.multivariate_normal.pdf(test_element, mean_virginica, cov_virginica)
#pdf_versicolor = scipy.stats.multivariate_normal.pdf(test_element, mean_versicolor, cov_versicolor)
#pdf_setosa = scipy.stats.multivariate_normal.pdf(test_element, mean_setosa, cov_setosa)
