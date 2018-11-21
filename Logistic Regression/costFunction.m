function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
sig =0;
[l,n] = size(theta);
if (l==1 && n==3)
    sig = X * theta';
else if (l == 3 && n==1)
        sig = X * theta;
    else
    end
end
h = sigmoid(sig);
a = log(h);
b = log (1-h);
J = (-y) .* (a) - (1-y) .* b;
J = (1/m) * sum(J);

for i = 1:length(theta)
    grad(i) = (1/m) * (sum((h - y) .* X(:,i)));
end


% =============================================================

end