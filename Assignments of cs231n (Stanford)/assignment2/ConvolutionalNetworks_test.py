# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:37:09 2017

@author: alan
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from cs231n.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
  
## Load the (preprocessed) CIFAR10 data.
#data = get_CIFAR10_data()
#for k, v in data.items():
#    print('%s: ' % k, v.shape)
  
################################################################################
## Convolution: Naive forward pass
################################################################################


################################################################################
## Forward pass check
#x_shape = (2, 3, 4, 4)
#w_shape = (3, 3, 4, 4)
#x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
#w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
#b = np.linspace(-0.1, 0.2, num=3)
#
#correct_out = np.array([[[[-0.08759809, -0.10987781],
#                           [-0.18387192, -0.2109216 ]],
#                          [[ 0.21027089,  0.21661097],
#                           [ 0.22847626,  0.23004637]],
#                          [[ 0.50813986,  0.54309974],
#                           [ 0.64082444,  0.67101435]]],
#                         [[[-0.98053589, -1.03143541],
#                           [-1.19128892, -1.24695841]],
#                          [[ 0.69108355,  0.66880383],
#                           [ 0.59480972,  0.56776003]],
#                          [[ 2.36270298,  2.36904306],
#                           [ 2.38090835,  2.38247847]]]])
#print(correct_out.shape)
#
#conv_param = {'stride': 2, 'pad': 1}
#out, _ = conv_forward_naive(x, w, b, conv_param)
#
## Compare your output to ours; difference should be around 2e-8
#print('Testing conv_forward_naive')
#print('difference: ', rel_error(out, correct_out))



################################################################################
## Convolution for image processing
#from scipy.misc import imread, imresize
#
#kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
## kitten is wide, and puppy is already square
#d = kitten.shape[1] - kitten.shape[0]
#kitten_cropped = kitten[:, d//2:-d//2, :]
#
#img_size = 200   # Make this smaller if it runs too slow
#x = np.zeros((2, 3, img_size, img_size))
#x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
#x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))
#
## Set up a convolutional weights holding 2 filters, each 3x3
#w = np.zeros((2, 3, 3, 3))
#
## The first filter converts the image to grayscale.
## Set up the red, green, and blue channels of the filter.
#w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
#w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
#w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
#
## Second filter detects horizontal edges in the blue channel.
#w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
#
## Vector of biases. We don't need any bias for the grayscale
## filter, but for the edge detection filter we want to add 128
## to each output so that nothing is negative.
#b = np.array([0, 128])
#
## Compute the result of convolving each input in x with each filter in w,
## offsetting by b, and storing the results in out.
#out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})
#
#def imshow_noax(img, normalize=True):
#    """ Tiny helper to show images as uint8 and remove axis labels """
#    if normalize:
#        img_max, img_min = np.max(img), np.min(img)
#        img = 255.0 * (img - img_min) / (img_max - img_min)
#    plt.imshow(img.astype('uint8'))
#    plt.gca().axis('off')
#
## Show the original images and the results of the conv operation
#plt.subplot(2, 3, 1)
#imshow_noax(puppy, normalize=False)
#plt.title('Original image')
#plt.subplot(2, 3, 2)
#imshow_noax(out[0, 0])
#plt.title('Grayscale')
#plt.subplot(2, 3, 3)
#imshow_noax(out[0, 1])
#plt.title('Edges')
#plt.subplot(2, 3, 4)
#imshow_noax(kitten_cropped, normalize=False)
#plt.subplot(2, 3, 5)
#imshow_noax(out[1, 0])
#plt.subplot(2, 3, 6)
#imshow_noax(out[1, 1])
#plt.show()


################################################################################
## Convolution: Naive backward pass
################################################################################
#np.random.seed(231)
#x = np.random.randn(4, 3, 5, 5)
#w = np.random.randn(2, 3, 3, 3)
#b = np.random.randn(2,)
#dout = np.random.randn(4, 2, 5, 5)
#conv_param = {'stride': 1, 'pad': 1}
#
#dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
#dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
#db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)
#
#out, cache = conv_forward_naive(x, w, b, conv_param)
#dx, dw, db = conv_backward_naive(dout, cache)
#
## Your errors should be around 1e-8'
#print('Testing conv_backward_naive function')
#print('dx error: ', rel_error(dx, dx_num))
#print('dw error: ', rel_error(dw, dw_num))
#print('db error: ', rel_error(db, db_num))


################################################################################
## Maxpooling forward pass
################################################################################
#x_shape = (2, 3, 4, 4)
#x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
#pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
#correct_out = np.array([[[[-0.26315789, -0.24842105],
#                          [-0.20421053, -0.18947368]],
#                         [[-0.14526316, -0.13052632],
#                          [-0.08631579, -0.07157895]],
#                         [[-0.02736842, -0.01263158],
#                          [ 0.03157895,  0.04631579]]],
#                        [[[ 0.09052632,  0.10526316],
#                          [ 0.14947368,  0.16421053]],
#                         [[ 0.20842105,  0.22315789],
#                          [ 0.26736842,  0.28210526]],
#                         [[ 0.32631579,  0.34105263],
#                          [ 0.38526316,  0.4       ]]]])
#print(correct_out.shape)
#out, _ = max_pool_forward_naive(x, pool_param)
#
## Compare your output with ours. Difference should be around 1e-8.
#print('Testing max_pool_forward_naive function:')
#print('difference: ', rel_error(out, correct_out))
#
#################################################################################
### Maxpooling backward pass
#################################################################################
#np.random.seed(231)
#x = np.random.randn(3, 2, 8, 8)
#dout = np.random.randn(3, 2, 4, 4)
#pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
#
#dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)
#
#out, cache = max_pool_forward_naive(x, pool_param)
#dx = max_pool_backward_naive(dout, cache)
#
## Your error should be around 1e-12
#print('Testing max_pool_backward_naive function:')
#print('dx error: ', rel_error(dx, dx_num))


################################################################################
## Fast Layers
################################################################################
#from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
#from time import time
#np.random.seed(231)
#x = np.random.randn(100, 3, 31, 31)
#w = np.random.randn(25, 3, 3, 3)
#b = np.random.randn(25,)
#dout = np.random.randn(100, 25, 16, 16)
#conv_param = {'stride': 2, 'pad': 1}
#
#t0 = time()
#out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
#t1 = time()
#out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
#t2 = time()
#
#print('Testing conv_forward_fast:')
#print('Naive: %fs' % (t1 - t0))
#print('Fast: %fs' % (t2 - t1))
#print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
#print('Difference: ', rel_error(out_naive, out_fast))
#
#t0 = time()
#dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
#t1 = time()
#dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
#t2 = time()
#
#print('\nTesting conv_backward_fast:')
#print('Naive: %fs' % (t1 - t0))
#print('Fast: %fs' % (t2 - t1))
#print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
#print('dx difference: ', rel_error(dx_naive, dx_fast))
#print('dw difference: ', rel_error(dw_naive, dw_fast))
#print('db difference: ', rel_error(db_naive, db_fast))


################################################################################
## Sanity check for loss
################################################################################
model = ThreeLayerConvNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization): ', loss)

model.reg = 0.5
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss)


################################################################################
## Sanity check for gradients
################################################################################
num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.0
num_classes = 10
np.random.seed(231)
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                          input_dim=input_dim, hidden_dim=7,
                          dtype=np.float64)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
    
################################################################################
## Sanity check for overfitting a small datsset
################################################################################
    
np.random.seed(231)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=1e-2)

solver = Solver(model, small_data,
                num_epochs=20, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 5e-4,
                },
                verbose=True, print_every=10)
solver.train()

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


################################################################################
## Train the 3-layerd ConvNet on CIFAR-10 for one epoch.
################################################################################
model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=100)
solver.train()

from cs231n.vis_utils import visualize_grid

grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()