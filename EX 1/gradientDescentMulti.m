function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

gd1 = 0;
gd2 = 0;
gd3 = 0;
for i = 1:m,
gd1 = gd1 + ((X(i, :)*theta) - (y(i,:)));
gd2 = gd2 + ((X(i, :)*theta) - (y(i,:)))*(X(i,2));
gd3 = gd3 + ((X(i, :)*theta) - (y(i,:)))*(X(i,3));
end;

theta(1) = theta(1) - (alpha/m)*gd1;
theta(2) = theta(2) - (alpha/m)*gd2;
theta(3) = theta(3) - (alpha/m)*gd3;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
