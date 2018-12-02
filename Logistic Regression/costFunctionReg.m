function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));
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

end
