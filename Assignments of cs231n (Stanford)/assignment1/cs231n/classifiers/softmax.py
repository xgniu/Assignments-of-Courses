import numpy as np
from random import shuffle
from numpy import matlib

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        label_onehot = np.zeros((1,num_classes))
        label_onehot[0,y[i]] = 1
        scores = X[i].dot(W)
        scores -= np.max(scores)
        scores = np.exp(scores)
        probs = scores / np.sum(scores)
        loss += -np.log(probs[y[i]])
        dW += X[i].reshape((1,-1)).T.dot(probs-label_onehot)
        
    loss /= num_train
    loss += reg * np.sum(W*W)
    
    dW /= num_train
    dW += 2*reg*W
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    label_onehot = np.zeros((num_train,num_classes))
    label_onehot[range(num_train),y] = 1
    scores = X.dot(W)
    row_max = np.max(scores,axis=1,keepdims=True)
    scores -= matlib.repmat(row_max,1,num_classes)
    scores = np.exp(scores)
    probs = scores / np.sum(scores,axis=1,keepdims=True)
    loss += -np.sum( np.log(probs[range(num_train),y]) )
    dW += X.T.dot(probs-label_onehot)
        
    loss /= num_train
    loss += reg * np.sum(W*W)
    
    dW /= num_train
    dW += 2*reg*W
    
    return loss, dW

