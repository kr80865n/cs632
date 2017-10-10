
# coding: utf-8

# # Part 1A) Coding Question

# In[2]:

# Importing Required Libraries

import numpy as np
from sklearn import datasets
import math


# ## Loading Data

# In[3]:

iris=datasets.load_iris()


# ## Differentiating between the label and the features used to predict

# In[4]:

iris_X=iris.data
iris_y=iris.target


# In[5]:

np.unique(iris_y)


# ## Spliting the data in training and test

# In[7]:

np.random.seed(123)
indices=np.random.permutation(len(iris_X))
iris_X_train=iris_X[indices[:110]]
iris_y_train=iris_y[indices[:110]]
iris_X_test=iris_X[indices[110:]]
iris_y_test=iris_y[indices[110:]]


# In[13]:

type(iris_X_train),type(iris_y_train),type(iris_X_test),type(iris_y_test)


# ## Create and Fit a nearest-neighbour classifier model and predicting Test data

# In[8]:

from collections import Counter

def train(X_train, y_train):
	return


def predict(iris_X_train, iris_y_train, iris_X_test, k):
    distances = []
    targets = []
    
    for i in range(len(iris_X_train)):
        distance = np.sqrt(np.sum(np.square(iris_X_test - iris_X_train[i, :])))
        #print("distance:" +str(distance))
        distances.append([distance, i])
        
    
    distances = sorted(distances)
    #print(distances)

    for i in range(k):
        index = distances[i][1]
        #print(index)
        targets.append(iris_y_train[index])
    #print(targets)
    return Counter(targets).most_common(1)[0][0]


def kNearestNeighbor(iris_X_train, iris_y_train, iris_X_test, y_pred, k):
    # train on the input data
    train(iris_X_train, iris_y_train)

    # loop over all observations
    for i in range(len(iris_X_test)):
        y_pred.append(predict(iris_X_train, iris_y_train, iris_X_test[i, :], k))

y_predictor = []
kNearestNeighbor(iris_X_train, iris_y_train, iris_X_test, y_predictor, 3)
y_predictor = np.asarray(y_predictor)
print(y_predictor)


# In[9]:

print ("Iris Y test: %s" %iris_y_test)


# ## Accuracy Calculator

# In[10]:

from sklearn.metrics import accuracy_score

print ("The accuracy of model is %s" %format(accuracy_score(iris_y_test,y_predictor)))

