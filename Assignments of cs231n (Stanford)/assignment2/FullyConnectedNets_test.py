import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from cs231n.layers import *

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    

################################################################################  
## Affine layer: forward   
## Test the affine_forward function
#num_inputs = 2
#input_shape = (4, 5, 6)
#output_dim = 3
#
#input_size = num_inputs * np.prod(input_shape)
#weight_size = output_dim * np.prod(input_shape)
#
#x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
#w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
#b = np.linspace(-0.3, 0.1, num=output_dim)
#
#out, _ = affine_forward(x, w, b)
#correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
#                        [ 3.25553199,  3.5141327,   3.77273342]])
#
## Compare your output with ours. The error should be around 1e-9.
#print('Testing affine_forward function:')
#print('difference: ', rel_error(out, correct_out))
#
################################################################################
## Affine layer: background
#np.random.seed(231)
#x = np.random.randn(10, 2, 3)
#w = np.random.randn(6, 5)
#b = np.random.randn(5)
#dout = np.random.randn(10, 5)
#
#dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
#dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
#db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
#
#_, cache = affine_forward(x, w, b)
#dx, dw, db = affine_backward(dout, cache)
#
## The error should be around 1e-10
#print('Testing affine_backward function:')
#print('dx error: ', rel_error(dx_num, dx))
#print('dw error: ', rel_error(dw_num, dw))
#print('db error: ', rel_error(db_num, db))
#
################################################################################
## ReLU layer: forward
## Test the relu_forward function
#x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
#
#out, _ = relu_forward(x)
#correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
#                        [ 0.,          0.,          0.04545455,  0.13636364,],
#                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])
#
## Compare your output with ours. The error should be around 5e-8
#print('Testing relu_forward function:')
#print('difference: ', rel_error(out, correct_out))
#
################################################################################
## ReLU: backward
#np.random.seed(231)
#x = np.random.randn(10, 10)
#dout = np.random.randn(*x.shape)
#
#dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
#
#_, cache = relu_forward(x)
#dx = relu_backward(dout, cache)
#
## The error should be around 3e-12
#print('Testing relu_backward function:')
#print('dx error: ', rel_error(dx_num, dx))


################################################################################
## Two-layer network
################################################################################

#np.random.seed(231)
#N, D, H, C = 3, 5, 50, 7 # H: hidden-layer dimension
#X = np.random.randn(N, D) # N: num_examples, D: feature dimension
#y = np.random.randint(C, size=N) # C: num_classes
#
#std = 1e-3
#model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
#
#print('Testing initialization ... ')
#W1_std = abs(model.params['W1'].std() - std)
#b1 = model.params['b1']
#W2_std = abs(model.params['W2'].std() - std)
#b2 = model.params['b2']
#assert W1_std < std / 10, 'First layer weights do not seem right'
#assert np.all(b1 == 0), 'First layer biases do not seem right'
#assert W2_std < std / 10, 'Second layer weights do not seem right'
#assert np.all(b2 == 0), 'Second layer biases do not seem right'
#
#print('Testing test-time forward pass ... ')
#model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
#model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
#model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
#model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
#X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
#scores = model.loss(X)
#correct_scores = np.asarray(
#  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
#   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
#   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
#scores_diff = np.abs(scores - correct_scores).sum()
#assert scores_diff < 1e-6, 'Problem with test-time forward pass'
#
#print('Testing training loss (no regularization)')
#y = np.asarray([0, 5, 1])
#loss, grads = model.loss(X, y)
#correct_loss = 3.4702243556
#assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
#
#model.reg = 1.0
#loss, grads = model.loss(X, y)
#correct_loss = 26.5948426952
#assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'
#
#for reg in [0.0, 0.7]:
#    print('Running numeric gradient check with reg = ', reg)
#    model.reg = reg
#    loss, grads = model.loss(X, y)
#
#    for name in sorted(grads):
#        f = lambda _: model.loss(X, y)[0]
#        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
#        print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))


################################################################################
# Solver 
################################################################################

# Load the (preprocessed) CIFAR10 data.
data = get_CIFAR10_data()
for k, v in list(data.items()):
    print(('%s: ' % k, v.shape))
#
#data_labeled = {
#  'X_train': data['X_train'],
#  'y_train': data['y_train'],
#  'X_val': data['X_val'],
#  'y_val': data['y_val']
#}
#X_train = data_labeled['X_train']
#input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
#model = TwoLayerNet(input_dim,hidden_dim=100,num_classes=10, reg=0.03)
#solver = Solver(model, data, update_rule='sgd', optim_config={ 'learning_rate': 8e-4, },
#                lr_decay=0.95, num_epochs=10, batch_size=100, print_every=100)
#solver.train()

# Run this cell to visualize training loss and train / val accuracy

#plt.subplot(2, 1, 1)
#plt.title('Training loss')
#plt.plot(solver.loss_history, 'o')
#plt.xlabel('Iteration')
#
#plt.subplot(2, 1, 2)
#plt.title('Accuracy')
#plt.plot(solver.train_acc_history, '-o', label='train')
#plt.plot(solver.val_acc_history, '-o', label='val')
#plt.plot([0.5] * len(solver.val_acc_history), 'k--')
#plt.xlabel('Epoch')
#plt.legend(loc='lower right')
#plt.gcf().set_size_inches(15, 12)
#plt.show()



################################################################################
## Multilayer network
################################################################################

## 1. Initial loss and gradient check
#np.random.seed(231)
#N, D, H1, H2, C = 2, 15, 20, 30, 10
#X = np.random.randn(N, D)
#y = np.random.randint(C, size=(N,))
#
#for reg in [0, 3.14]:
#    print('Running check with reg = ', reg)
#    model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
#                            reg=reg, weight_scale=5e-2, dtype=np.float64)
#
#    loss, grads = model.loss(X, y)
#    print('Initial loss: ', loss)
#
#    for name in sorted(grads):
#        f = lambda _: model.loss(X, y)[0]
#        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#        print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#        



################################################################################        
## 2. Use a 3-layer net to overfit 50 training examples
#num_train = 50
#small_data = {
#  'X_train': data['X_train'][:num_train],
#  'y_train': data['y_train'][:num_train],
#  'X_val': data['X_val'],
#  'y_val': data['y_val'],
#}
#
#weight_scale = 1e-2
#learning_rate = 1e-2
#model = FullyConnectedNet([100, 100], weight_scale=weight_scale, dtype=np.float64)
#solver = Solver(model, small_data,
#                print_every=10, num_epochs=20, batch_size=25,
#                update_rule='sgd',
#                optim_config={ 'learning_rate': learning_rate, }
#         )
#solver.train()
#
#plt.plot(solver.loss_history, 'o')
#plt.title('Training loss history')
#plt.xlabel('Iteration')
#plt.ylabel('Training loss')
#plt.show()

################################################################################
## 3. Use a 5-layer net to overfit 50 training examples
#num_train = 50
#small_data = {
#  'X_train': data['X_train'][:num_train],
#  'y_train': data['y_train'][:num_train],
#  'X_val': data['X_val'],
#  'y_val': data['y_val'],
#}
#
#learning_rate = 2e-2
#weight_scale = 1e-2
#model = FullyConnectedNet([100, 100, 100, 100],
#                weight_scale=weight_scale, dtype=np.float64)
#solver = Solver(model, small_data,
#                print_every=10, num_epochs=20, batch_size=25,
#                update_rule='sgd',
#                optim_config={'learning_rate': learning_rate,} )
#solver.train()
#
#plt.plot(solver.loss_history, 'o')
#plt.title('Training loss history')
#plt.xlabel('Iteration')
#plt.ylabel('Training loss')
#plt.show()

################################################################################
## Update Rules
################################################################################

################################################################################
# 1. SGD+Momentum vs. SGD
#num_train = 4000
#small_data = {
#  'X_train': data['X_train'][:num_train],
#  'y_train': data['y_train'][:num_train],
#  'X_val': data['X_val'],
#  'y_val': data['y_val'],
#}
#
#solvers = {}
#
#for update_rule in ['sgd', 'sgd_momentum']:
#    print('running with ', update_rule)
#    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)
#
#    solver = Solver(model, small_data,
#                  num_epochs=5, batch_size=100,
#                  update_rule=update_rule,
#                  optim_config={ 'learning_rate': 1e-2,},
#                  verbose=True)
#    solvers[update_rule] = solver
#    solver.train()
#    print()
#
#plt.subplot(3, 1, 1)
#plt.title('Training loss')
#plt.xlabel('Iteration')
#
#plt.subplot(3, 1, 2)
#plt.title('Training accuracy')
#plt.xlabel('Epoch')
#
#plt.subplot(3, 1, 3)
#plt.title('Validation accuracy')
#plt.xlabel('Epoch')
#
#for update_rule, solver in list(solvers.items()):
#    plt.subplot(3, 1, 1)
#    plt.plot(solver.loss_history, 'o', label=update_rule)
#  
#    plt.subplot(3, 1, 2)
#    plt.plot(solver.train_acc_history, '-o', label=update_rule)
#
#    plt.subplot(3, 1, 3)
#    plt.plot(solver.val_acc_history, '-o', label=update_rule)
#  
#for i in [1, 2, 3]:
#    plt.subplot(3, 1, i)
#    plt.legend(loc='upper center', ncol=4)
#plt.gcf().set_size_inches(15, 15)
#plt.show()


################################################################################
## Adam, RMSprop, SGD+Momentum, SGD
#learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
#for update_rule in ['adam', 'rmsprop']:
#    print('running with ', update_rule)
#    model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)
#
#    solver = Solver(model, small_data,
#                  num_epochs=5, batch_size=100,
#                  update_rule=update_rule,
#                  optim_config={ 'learning_rate': learning_rates[update_rule] },
#                  verbose=True)
#    solvers[update_rule] = solver
#    solver.train()
#    print()
#
#plt.subplot(3, 1, 1)
#plt.title('Training loss')
#plt.xlabel('Iteration')
#
#plt.subplot(3, 1, 2)
#plt.title('Training accuracy')
#plt.xlabel('Epoch')
#
#plt.subplot(3, 1, 3)
#plt.title('Validation accuracy')
#plt.xlabel('Epoch')
#
#for update_rule, solver in list(solvers.items()):
#    plt.subplot(3, 1, 1)
#    plt.plot(solver.loss_history, 'o', label=update_rule)
#  
#    plt.subplot(3, 1, 2)
#    plt.plot(solver.train_acc_history, '-o', label=update_rule)
#
#    plt.subplot(3, 1, 3)
#    plt.plot(solver.val_acc_history, '-o', label=update_rule)
#  
#for i in [1, 2, 3]:
#    plt.subplot(3, 1, i)
#    plt.legend(loc='upper center', ncol=4)
#plt.gcf().set_size_inches(15, 15)
#plt.show()


data = get_CIFAR10_data()
num_train = 4000
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

#best_model=None
#val_acc = 0
#
#dropouts = [0.5,0.6]
#learning_rates = [1e-5,5e-5,1e-4,2e-4]
#weight_scale = 2e-2
#for do in dropouts:
#    for lr in learning_rates:
#        model = FullyConnectedNet([100, 100], weight_scale=weight_scale, reg = 1e-2,
#                                  dropout = do, use_batchnorm=True, dtype=np.float64)
#        solver = Solver( model, small_data, verbose = False,
#                print_every=10, num_epochs=20, batch_size=25,
#                update_rule='adam',
#                optim_config={ 'learning_rate': lr, } )
#        solver.train()
#        print('do=%f, lr=%f, val_acc=%f' %(do,lr,solver.val_acc_history[-1]))
#        if solver.val_acc_history[-1] > val_acc:        
#            best_model = model
best_model=None
val_acc = 0
regs = [1e-3,5e-3,1e-2,5e-2]
dropout = 0.5
learning_rate = 2e-4
weight_scale = 2e-2
for reg in regs:
    model = FullyConnectedNet([100, 100], weight_scale=weight_scale, reg = reg,
                                  dropout=dropout, use_batchnorm=True, dtype=np.float64)
    solver = Solver( model, small_data, verbose = False,
                print_every=10, num_epochs=20, batch_size=50,
                update_rule='adam',
                optim_config={ 'learning_rate': learning_rate, } )
    solver.train()
    print('do=%f, lr=%f, val_acc=%f' %(dropout,learning_rate,solver.val_acc_history[-1]))
    if solver.val_acc_history[-1] > val_acc:        
        best_model = model

y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())