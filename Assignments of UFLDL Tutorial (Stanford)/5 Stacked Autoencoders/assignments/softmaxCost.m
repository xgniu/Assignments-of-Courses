function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize  - the size N of the input vector
% lambda     - weight decay parameter
% data       - the N x M input matrix, where each column data(:, i) corresponds to a single sample
% labels     - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = theta * data;
M = bsxfun( @minus, M, max(M, [], 1) );
H = exp(M);
H = bsxfun( @rdivide, H, sum(H) );

cost = groundTruth .* log(H);
cost = - sum(cost(:)) / numCases + lambda*norm(theta,'fro')^2 / 2;

thetagrad = - (groundTruth - H) * data' / numCases + lambda*theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

