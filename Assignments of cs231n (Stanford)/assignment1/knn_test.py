# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:43:25 2017

@author: alan
"""
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.array_split(X_train,num_folds)
y_train_folds = np.array_split(y_train,num_folds)

k_to_accuracies = {}
for i in range(len(k_choices)):
    accuracy=[]
    for j in range(num_folds):
        X_train_ = np.reshape(np.asarray( X_train_folds[:j]+X_train_folds[j+1:] ),(-1,3072))
        y_train_ = np.reshape(np.asarray( y_train_folds[:j]+y_train_folds[j+1:] ),(4000,-1))
        X_test_ = np.asarray( X_train_folds[j] )
        y_test_ = np.asarray( y_train_folds[j] )
        
        from cs231n.classifiers import KNearestNeighbor
        classifier = KNearestNeighbor()
        classifier.train(X_train_, y_train_)
        y_test_pred = classifier.predict(X_test_, k=k_choices[i], num_loops=0)

        accuracy.append( np.sum(y_test_pred == y_test_) / y_test_.shape[0] )
    k_to_accuracies[k_choices[i]] = accuracy

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
        
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()