# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:51:13 2017

@author: alan
"""

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

###############################################################################
# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#num_classes = len(classes)
#samples_per_class = 7
#for y, cls in enumerate(classes):
#    idxs = np.flatnonzero(y_train == y)
#    idxs = np.random.choice(idxs, samples_per_class, replace=False)
#    for i, idx in enumerate(idxs):
#        plt_idx = i * num_classes + y + 1
#        plt.subplot(samples_per_class, num_classes, plt_idx)
#        plt.imshow(X_train[idx].astype('uint8'))
#        plt.axis('off')
#        if i == 0:
#            plt.title(cls)
#plt.show()

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


###############################################################################
# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

# Prepocessing: subtract the mean image (calculated from train_data)
mean_image = np.mean(X_train,axis=0)
print(mean_image[:10])
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))
plt.show()

X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# Prepocessing: append the bias dimension of ones
X_train = np.hstack([X_train, np.ones((X_train.shape[0],1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0],1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0],1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0],1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)


################################################################################
#from cs231n.classifiers.linear_svm import svm_loss_naive
#import time
#
#W = np.random.randn(3073, 10) * 0.0001 

## Numerically compute the gradient along several randomly chosen dimensions, and
## compare them with your analytically computed gradient. The numbers should match
## almost exactly along all dimensions.
#from cs231n.gradient_check import grad_check_sparse
#
#loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
#f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
#grad_numerical = grad_check_sparse(f, W, grad)

################################################################################
## Next implement the function svm_loss_vectorized; 
## for now only compute the loss;
#tic = time.time()
#loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
#toc = time.time()
#print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))
#
#from cs231n.classifiers.linear_svm import svm_loss_vectorized
#tic = time.time()
#loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
#toc = time.time()
#print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))
#
## The losses should match but your vectorized implementation should be much faster.
#print('difference: %f' % (loss_naive - loss_vectorized))
#
################################################################################
## Complete the implementation of svm_loss_vectorized, and compute the gradient
## of the loss function in a vectorized way.
#
## The naive implementation and the vectorized implementation should match, but
## the vectorized version should still be much faster.
#tic = time.time()
#_, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
#toc = time.time()
#print('Naive loss and gradient: computed in %fs' % (toc - tic))
#
#tic = time.time()
#_, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
#toc = time.time()
#print('Vectorized loss and gradient: computed in %fs' % (toc - tic))
#
## The loss is a single number, so it is easy to compare the values computed
## by the two implementations. The gradient on the other hand is a matrix, so
## we use the Frobenius norm to compare them.
#difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
#print('difference: %f' % difference)

################################################################################
## Use Cross validation to tune hyperparameters
################################################################################
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.4 on the validation set.
learning_rates = [1e-8,1e-7, 5e-7, 1e-6]
regularization_strengths = [5e3,1e4, 2.5e4]
from cs231n.classifiers import LinearSVM

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=1500, verbose=False)
        y_train_pred = svm.predict(X_train)
        tr_acc = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val)
        val_acc = np.mean(y_val == y_val_pred)        
        print('lr: %f, reg: %f, training accuracy: %f, validation accuracy: %f' % (lr,reg,tr_acc,val_acc,))
        results[(lr,reg)] = (tr_acc, val_acc)
        if val_acc>best_val:
            best_val = val_acc
            best_svm = svm

# Print out results.
#for lr, reg in sorted(results):
#    train_accuracy, val_accuracy = results[(lr, reg)]
#    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
      
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])