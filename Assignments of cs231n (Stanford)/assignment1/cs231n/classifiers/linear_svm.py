import numpy as np
from random import shuffle
from numpy import matlib

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        dWi = np.zeros(W.shape)
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):                
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dWi[:,j] = X[i]
        for j in range(num_classes):                
            if j != y[i]:
                dWi[:,y[i]] -= dWi[:,j]
        dW += dWi

    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2*reg*W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    scores = X.dot(W)
    tmp1 = scores[range(num_train),y] - 1
    scores -= matlib.repmat(tmp1,num_classes,1).T
    scores[range(num_train),y] = 0
    
    tmp2 = scores
    tmp2[tmp2<0] = 0
    tmp2[tmp2>0] = 1
    tmp2[range(num_train),y] = 0
    tmp3 = -np.sum(tmp2,axis=1)
    tmp2[range(num_train),y] = tmp3
    
    dW = np.dot(X.T,tmp2)
    dW /= num_train
    dW += 2*reg*W
    
    scores[scores<0] = 0
    loss = np.sum(scores)
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    return loss, dW
