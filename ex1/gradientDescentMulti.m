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

   J1=0;
for i = 1:m
    J1 = J1 + ( (((theta(1,1) * X(i,1)) + (theta(2,1) * X(i,2))+ (theta(3,1) * X(i,3))) - y(i,:)) * (X(i,1)) );
end
     J2=0;
for i = 1:m
    J2 = J2 + ( (((theta(1,1) * X(i,1)) + (theta(2,1) * X(i,2))+ (theta(3,1) * X(i,3))) - y(i,:)) * (X(i,2)) );
end
 J3=0;
for i = 1:m
    J3 = J3 + ( (((theta(1,1) * X(i,1)) + (theta(2,1) * X(i,2))+ (theta(3,1) * X(i,3))) - y(i,:)) * (X(i,3)) );
end
    theta(1,1) = theta(1,1) - alpha * (1/m) * J1;
    theta(2,1) = theta(2,1) - alpha * (1/m) * J2;
    theta(3,1) = theta(3,1) - alpha * (1/m) * J3;










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
