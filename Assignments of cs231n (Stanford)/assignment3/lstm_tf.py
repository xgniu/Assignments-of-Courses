#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:30:04 2017

@author: alan
"""
import time, os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cs231n.layers import *
from cs231n.rnn_layers import *
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url


class CaptioningLSTM(object):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, features_dim=512, captions_dim=17, wordvec_dim=256, hidden_dim=512):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Make a minibatch of training data
        self.batch_size = 25
        #tf.get_variable_scope().reuse_variables()
        tf.reset_default_graph()
        
        with tf.variable_scope('input') as scope:
            self.input_features = tf.placeholder(tf.float32,shape=[self.batch_size,features_dim])
            self.input_captions = tf.placeholder(tf.int32,shape=[self.batch_size,captions_dim])

        # Initialize word vectors
        with tf.variable_scope('word2vec') as scope:
            #self.W_embed = tf.get_variable('W_embed',[vocab_size, wordvec_dim],dtype=tf.float32)
            self.W_embed = tf.Variable(tf.random_uniform([vocab_size, wordvec_dim], -1.0, 1.0))

        # Initialize CNN -> hidden state projection parameters
        with tf.variable_scope('proj') as scope:
            self.W_proj = tf.get_variable('W_proj',[features_dim, hidden_dim],dtype=tf.float32)
            self.b_proj = tf.get_variable('b_proj',[hidden_dim],dtype=tf.float32)

        # Initialize parameters for the RNN
        with tf.variable_scope('lstm') as scope:
            self.Wx = tf.get_variable('Wx',[wordvec_dim, 4*hidden_dim],dtype=tf.float32)
            self.Wh = tf.get_variable('Wh',[hidden_dim, 4*hidden_dim],dtype=tf.float32)
            self.b = tf.get_variable('b',[4*hidden_dim])

        # Initialize output to vocab weights
        with tf.variable_scope('output') as scope:
            self.W_vocab = tf.get_variable('W_vocab',[hidden_dim, vocab_size],dtype=tf.float32)
            self.b_vocab = tf.get_variable('b_vocab',[vocab_size,1],dtype=tf.float32)

        captions_in = self.input_captions[:, :-1]
        captions_out = self.input_captions[:, 1:]

        mask = (captions_out != self._null)
        
        h0 = tf.matmul(self.input_features,self.W_proj) + self.b_proj       
        wvec_in, _ = tf.nn.embedding_lookup(self.W_embed,captions_in)
        h,_ = lstm_forward(wvec_in,h0,self.Wx,self.Wh,self.b)       
        scores,_ = temporal_affine_forward(h,self.W_vocab,self.b_vocab)
        self.loss, _ = temporal_softmax_loss(scores,captions_out,mask)
        
        self.global_step = tf.Variable(0,name='slobal_step',trainable=False)
        self.lr = tf.train.exponential_decay(5e-3,self.global_step,1e10,0.9)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),5)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars),self.global_step)

    def train(self, data, num_epochs):
        """
        Train the model by tensorflow.
        """
        # Make a minibatch of training data
        self.batch_size = 25
        loss_history = []
        print('Start Training')              
        with tf.Session() as sess:
            num_train = data['train_captions'].shape[0]
            iterations_per_epoch = max(num_train // self.batch_size, 1)
            num_iterations = num_epochs * iterations_per_epoch
            sess.run(tf.global_variables_initializer())
            for t in range(num_iterations):
                minibatch = sample_coco_minibatch(data, self.batch_size, split='train')
                captions, features, urls = minibatch
                _, loss_t = sess.run([self.train_op,self.loss],feed_dict= \
                                     {self.input_features:features,self.input_captions:captions})
                loss_history.append(loss_t)
                if t % 1000 ==0:
                    print('Iteration: %d, loss: %f' %(t,loss_t))
        return loss_history

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        prev_h = np.dot(features,self.W_proj) + self.b_proj
        prev_c = np.zeros_like(prev_h)
        prev_word_idx = [self._start] * N # word idx, (N,)
        for t in range(max_length):
            prev_word_vec, _ = word_embedding_forward(prev_word_idx,self.W_embed) # (N,W)
            next_h,next_c,_ = lstm_step_forward(prev_word_vec,prev_h,prev_c,self.Wx,self.Wh,self.b)
            scores = np.dot(next_h,self.W_vocab) + self.b_vocab
            next_word_idx = np.argmax(scores,axis=1)    
            captions[:,t] = next_word_idx
            prev_h = next_h
            prev_c = next_c
            prev_word_idx = next_word_idx

        return captions


# Load COCO data from disk; this returns a dictionary
# We'll work with dimensionality-reduced features for this notebook, but feel
# free to experiment with the original features by changing the flag below.
data = load_coco_data(max_train=100,pca_features=True)

# Print out all the keys and values from the data dictionary
for k, v in data.items():
    if type(v) == np.ndarray:
        print(k, type(v), v.shape, v.dtype)
    else:
        print(k, type(v), len(v))
    
        
lstm_model = CaptioningLSTM(word_to_idx=data['word_to_idx'], features_dim=data['train_features'].shape[1],
          captions_dim = data['train_captions'].shape[1], hidden_dim=512, wordvec_dim=256)

num_epochs = 1
loss_history = lstm_model.train(data,num_epochs)

# Plot the training losses
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.show()