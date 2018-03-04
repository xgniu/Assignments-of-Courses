visibleSize = 8*8;      % number of input units 
hiddenSize = 25;        % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
lambda = 0.0001;        % weight decay parameter       
beta = 3;               % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: sampleIMAGES

patches = sampleIMAGES;

% display a random sample of 200 patches from the dataset
display_network(patches(:,randi(size(patches,2),200,1)),8);

% Obtain random parameters theta
theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: sparseAutoencoderCost
% costs: squared error cost, weight decay term, sparsity penalty

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);

%%======================================================================
%% STEP 3: Gradient Checking

% First, lets make sure your numerical gradient computation is correct for a
% simple function.  After you have implemented computeNumericalGradient.m,
% run the following: 
% checkNumericalGradient();

% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.  
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
            hiddenSize, lambda, sparsityParam, beta, patches), theta);

% Use this to visually compare the gradients side by side
% disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are usually less than 1e-9.

%%======================================================================
% STEP 4: train sparse autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; 
% Here, we use L-BFGS to optimize our cost function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the function value and the gradient.
% In our problem, sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, beta, patches), ...
                                   theta, options);

%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 


