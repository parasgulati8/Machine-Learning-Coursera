function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sig =0;
[l,n] = size(theta);
[u, v] = size(X);

if (n==v)
    sig = X * theta';
else if (l==v)
    sig = X * theta;
    else
    end
end

h = sigmoid(sig);
a = log(h);
b = log (1-h);
J = (-y) .* (a) - (1-y) .* b;
J = (1/m) * sum(J);

regParameter = theta;
regParameter(1) = 0;
for k = 2:length(theta)
regParameter(k) = regParameter(k)^2;
end
regParameter = sum(regParameter);
regParameter = (lambda/(2*m))*regParameter;
J = J + regParameter;

 grad(1) = (1/m) * (sum((h - y) .* X(:,1)));

for i = 2:v
    grad(i) = (1/m) * (sum((h - y) .* X(:,i))) + sum((lambda/m) .* theta(i));
end
% =============================================================

end
