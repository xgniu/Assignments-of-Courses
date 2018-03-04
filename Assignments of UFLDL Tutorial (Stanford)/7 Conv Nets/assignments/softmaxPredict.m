function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix

M = theta * data;
M = bsxfun( @minus, M, max(M, [], 1) );
H = exp(M);
H = bsxfun( @rdivide, H, sum(H) );
[~, pred] = max(H,[],1);

end

