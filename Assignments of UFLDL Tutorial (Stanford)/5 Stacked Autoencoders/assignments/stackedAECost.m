function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% inputSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

a1 = data;
z2 = stack{1}.w * a1 + repmat(stack{1}.b,1,m);
a2 = sigmoid(z2);
z3 = stack{2}.w * a2 + repmat(stack{2}.b,1,m);
a3 = sigmoid(z3);

M = softmaxTheta * a3;
M = bsxfun( @minus, M, max(M, [], 1) );
H = exp(M);
H = bsxfun( @rdivide, H, sum(H) );

cost = groundTruth .* log(H);
cost = - sum(cost(:)) / m + lambda*norm(softmaxTheta,'fro')^2 / 2;

softmaxThetaGrad = - (groundTruth - H) * a3' / m + lambda*softmaxTheta;

delta3 = - softmaxTheta' * (groundTruth - H) .* sigmoid_prime(z3);
delta2 = ( stack{2}.w' * delta3 ) .* sigmoid_prime(z2);

stackgrad{2}.w = delta3 * a2' /m;
stackgrad{2}.b = sum(delta3,2)/m;

stackgrad{1}.w = delta2 * a1'/m;
stackgrad{1}.b = sum(delta2,2)/m;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function sigm_prime = sigmoid_prime(x)
    sigm_prime = sigmoid(x) .* ( 1 - sigmoid(x) );
end
