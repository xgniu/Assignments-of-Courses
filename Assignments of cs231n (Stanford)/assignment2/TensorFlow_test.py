#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:35:27 2017

@author: alan
"""

import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

#################################################################################
### Part 1: Get Started
#################################################################################
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000,num_validation=1000,num_test=10000):
    cifar10_path = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_path)
    
    mask = range(num_training,num_training+num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    mean_image = np.mean(X_train,axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    return X_train,y_train,X_val,y_val,X_test,y_test


X_train,y_train,X_val,y_val,X_test,y_test = get_CIFAR10_data()
print('Train data shape: ',X_train.shape)
print('Train label shape: ',y_train.shape)
print('Validation data shape: ',X_val.shape)
print('Validation label shape: ',y_val.shape)
print('Test data shape: ',X_test.shape)
print('Test label shape: ',y_test.shape)
#
#
#tf.reset_default_graph()
#
#X = tf.placeholder(tf.float32,[None,32,32,3])
#y = tf.placeholder(tf.int64,[None])
#is_training = tf.placeholder(tf.bool)
#
#def simple_model(X):
#    w_conv1 = tf.get_variable('w_conv1',shape=[7,7,3,32])
#    b_conv1 = tf.get_variable('b_conv1',shape=[32])
#    w1 = tf.get_variable('w1',shape=[5408,10])
#    b1 = tf.get_variable('b1',shape=[10])
#    
#    a1 = tf.nn.conv2d(X,w_conv1,strides = [1,2,2,1],padding='VALID') + b_conv1
#    h1 = tf.nn.relu(a1)
#    h1_flat = tf.reshape(h1,[-1,5408])
#    y_out = tf.matmul(h1_flat,w1) + b1
#    return y_out
#
#y_out = simple_model(X)
#
#correct = tf.equal(tf.argmax(y_out,1),y)
#accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
#
#total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
#mean_loss = tf.reduce_mean(total_loss)
#
#optimizer = tf.train.AdamOptimizer(5e-4)
#train_op = optimizer.minimize(mean_loss)
#
#
#def run_model(session,Xd,yd,epochs=1,batch_size=64,print_every=100,training=None,plot_losses=False):
#    
#    train_indices = np.arange(Xd.shape[0])
#    np.random.shuffle( train_indices )
#    is_training_now = ( training is not None )
#    
#    fetches = [correct, mean_loss]
#    if is_training_now:
#        fetches.append(train_op)
#    else:
#        fetches.append(accuracy)
#    
#    iter_cnt = 0
#    for i in range(epochs):
#        correct_cnt = 0
#        losses = []
#        for j in range( int(math.ceil(Xd.shape[0]/batch_size)) ):
#            start_idx = (j*batch_size) % Xd.shape[0]
#            idx = train_indices[ start_idx : start_idx+batch_size]
#            actual_batch_size = yd[idx].shape[0]
#            
#            feed_dict = {X:Xd[idx,:], y:yd[idx], is_training:is_training_now}
#            corr, loss, _ = session.run(fetches,feed_dict)
#            losses.append(loss*actual_batch_size)
#            correct_cnt += np.sum(corr)
#            
#            if is_training_now and (iter_cnt % print_every)==0:
#                print('Iter: %d, minibatch loss: %f, accuracy: %f.'%(iter_cnt,loss,np.sum(corr)/actual_batch_size))
#            iter_cnt += 1
#        
#        total_accuracy = correct_cnt / Xd.shape[0]
#        total_loss = np.sum(losses) / Xd.shape[0]
#        print('Epoch %d, overall loss: %f, accuracy: %f.'%(i+1,total_loss,total_accuracy))
#        
#        if plot_losses:
#            plt.plot(losses); plt.grid(True); plt.title('Epoch {} loss'.format(i+1));
#            plt.xlabel('minibatch number'); plt.ylabel('minibatch loss'); plt.show()
#    return total_loss, total_accuracy
#
#
#with tf.Session() as sess:
#    with tf.device('/cpu:0'):
#        sess.run(tf.global_variables_initializer())
#        print('================  Training  ==============')
#        run_model(sess,X_train,y_train,training=True)
#        print('================ Validation ==============')
#        run_model(sess,X_val,y_val)
        
################################################################################
## Part 2: Specify a model
################################################################################

# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

# define model
def complex_model(X,is_training):
    reg = 1e-3
    with tf.variable_scope('conv1') as scope:
        w1_1 = tf.get_variable('w1_1',shape=[7,7,3,64])
        b1_1 = tf.get_variable('b1_1',shape=[64])
        l2_loss = reg * tf.nn.l2_loss(w1_1)
        conv1_1 = tf.nn.conv2d(X,w1_1,strides=[1,1,1,1],padding='SAME') + b1_1      
        conv1_1_relu = tf.nn.relu(conv1_1)
        
        w1_2 = tf.get_variable('w1_2',shape=[3,3,64,64])
        b1_2 = tf.get_variable('b1_2',shape=[64])
        l2_loss = reg * tf.nn.l2_loss(w1_2)
        conv1_2 = tf.nn.conv2d(conv1_1_relu,w1_2,strides=[1,1,1,1],padding='SAME') + b1_2        
        conv1_2_relu = tf.nn.relu(conv1_2)
        
        conv1_2_relu_maxp = tf.nn.max_pool(conv1_2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #conv1_relu_maxp_lrn = tf.nn.local_response_normalization(conv1_relu_maxp,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
        conv1_2_relu_maxp_bn = tf.layers.batch_normalization(conv1_2_relu_maxp,axis=3)
        
    with tf.variable_scope('conv2'):
        w2_1 = tf.get_variable('w2_1',shape=[3,3,64,128])
        b2_1 = tf.get_variable('b2_1',shape=[128])
        l2_loss += reg * tf.nn.l2_loss(w2_1)
        conv2_1 = tf.nn.conv2d(conv1_2_relu_maxp_bn,w2_1,strides=[1,1,1,1],padding='SAME') + b2_1      
        conv2_1_relu = tf.nn.relu(conv2_1)
        
        w2_2 = tf.get_variable('w2_2',shape=[3,3,128,128])
        b2_2 = tf.get_variable('b2_2',shape=[128])
        l2_loss = reg * tf.nn.l2_loss(w2_2)
        conv2_2 = tf.nn.conv2d(conv2_1_relu,w2_2,strides=[1,1,1,1],padding='SAME') + b2_2        
        conv2_2_relu = tf.nn.relu(conv2_2)
        
        conv2_2_relu_maxp = tf.nn.max_pool(conv2_2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #conv1_relu_maxp_lrn = tf.nn.local_response_normalization(conv1_relu_maxp,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
        conv2_2_relu_maxp_bn = tf.layers.batch_normalization(conv2_2_relu_maxp,axis=3)
    
    with tf.variable_scope('affine2') as scope:
        #print(conv1_relu_maxp_lrn.shape) # np.prod(conv1_relu_maxp_lrn.shape[1:])
        #reshape = tf.reshape( conv1_relu_maxp_lrn, [-1,8192] )
        reshape = tf.reshape( conv2_2_relu_maxp_bn, [-1,8192] )
        w = tf.get_variable('w',shape=[8192,1024])
        b = tf.get_variable('b',shape=[1024])
        l2_loss += reg * tf.nn.l2_loss(w)
        affine2 = tf.matmul(reshape,w) + b
        affine2_relu = tf.nn.relu(affine2)
        affine2_relu_dropout = tf.layers.dropout(affine2_relu,is_training)
    
    with tf.variable_scope('affine3') as scope:
        w = tf.get_variable('w',shape=[1024,10])
        b = tf.get_variable('b',shape=[10])
        l2_loss += reg * tf.nn.l2_loss(w)
        affine3 = tf.matmul(affine2_relu_dropout,w) + b
    
    return affine3, l2_loss

y_out,l2_loss = complex_model(X,is_training)
correct = tf.equal(tf.argmax(y_out,1),y)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

mean_loss = l2_loss + tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_out))
global_step = tf.Variable(0,name='global_step',trainable=False)
lr = tf.train.exponential_decay(1e-3,global_step,1e3,0.9)
optimizer = tf.train.RMSPropOptimizer(lr)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss,global_step=global_step)


def run_model(session,Xd,yd,epochs=1,batch_size=64,print_every=200,training=None,plot_losses=False):
    
    train_indices = np.arange(Xd.shape[0])
    np.random.shuffle( train_indices )
    is_training_now = ( training is not None )
    
    fetches = [correct, mean_loss]
    if is_training_now:
        fetches.append(train_step)
    else:
        fetches.append(accuracy)
    
    iter_cnt = 0
    loss_history = []
    for i in range(epochs):
        epoch_correct_cnt = 0
        epoch_loss = 0      
        for j in range( int(math.ceil(Xd.shape[0]/batch_size)) ):
            start_idx = (j*batch_size) % Xd.shape[0]
            idx = train_indices[ start_idx : start_idx+batch_size]
            actual_batch_size = yd[idx].shape[0]
            
            feed_dict = {X:Xd[idx,:], y:yd[idx], is_training:is_training_now}
            corr, loss, _ = session.run(fetches,feed_dict)
            loss_history.append(loss*actual_batch_size)
            epoch_correct_cnt += np.sum(corr)
            epoch_loss += loss*actual_batch_size
            
            if is_training_now and (iter_cnt % print_every)==0:
                print('Iter: %d, minibatch loss: %f, accuracy: %f.'%(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        
        total_accuracy = epoch_correct_cnt / Xd.shape[0]
        total_loss = epoch_loss / Xd.shape[0]
        print('Epoch %d, overall loss: %f, accuracy: %f.'%(i+1,total_loss,total_accuracy))
    
    if plot_losses:
        plt.plot(loss_history); plt.grid(True); plt.title('Epoch {} loss'.format(i+1));
        plt.xlabel('minibatch number'); plt.ylabel('minibatch loss'); plt.show()
    return total_loss, total_accuracy


with tf.Session() as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        print('================  Training  ==============')
        run_model(sess,X_train,y_train,training=True,epochs=5,plot_losses=True)
        # Epoch 5, overall loss: 0.803008, accuracy: 0.803714.
        print('================ Validation ==============')
        run_model(sess,X_val,y_val)
        # Epoch 1, overall loss: 1.404320, accuracy: 0.719000.