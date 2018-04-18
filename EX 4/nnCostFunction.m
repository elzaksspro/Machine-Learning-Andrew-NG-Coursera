function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Create a matrix y by recoding labels to vectors containing only 0 and 1.
ymatrix = [1:num_labels] == y;
X = [ones(m,1) X];

a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2,1),1) a2];

% Calculating our predictions
a3 = sigmoid(a2*Theta2');


% Calculating un-regularized cost function
for i=1:m,
  J = J + sum(ymatrix(i,:).*log(a3(i,:)) + (1 - ymatrix(i,:)).*log(1 - a3(i,:)));
end;
J = -J/m;

% Calculating the regularization part of the cost function
theta_cost = 0;
for i = 2:size(Theta1,2),
  theta_cost = theta_cost + sum(Theta1(:,i).^2);
end;

for i = 2:size(Theta2,2),
  theta_cost = theta_cost + sum(Theta2(:,i).^2);
end;

% Final cost function value with regularization
J = J + (lambda/(2*m))*(theta_cost);

% Errors in the output layer
d3 = a3 - ymatrix; % This has dimension (m X k)

d2 = (d3*Theta2(:,2:end)).*(sigmoidGradient(X*Theta1')); % We do not use the bias unit in Theta2


% Calculating the delta(Theta_grad) for each layer
% These give us the unregularized gradients of the cost function
Theta1_grad = (d2'*X)/m;
Theta2_grad = (d3'*a2)/m;

% Here, we change regularize the values of the parameters except the first columns which
% includes the bias unit
Theta1(:,2:end) = [(lambda/m)*Theta1(:,2:end)];
Theta2(:,2:end) = [(lambda/m)*Theta2(:,2:end)];

% Finally, we get the gradients using regularization
Theta1_grad = [Theta1_grad(:,1) (Theta1_grad(:,2:end) + Theta1(:,2:end))];
Theta2_grad = [Theta2_grad(:,1) (Theta2_grad(:,2:end) + Theta2(:,2:end))];

% We want to keep the first column of our unregularized gradients unchanged.
% For the rest of the columns, we will add the unregularized gradient and the
% regularized theta parameters.

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
