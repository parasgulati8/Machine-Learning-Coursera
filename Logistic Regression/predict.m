function p = predict(theta, X)
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples
p = zeros(m, 1);

[l,n] = size(theta);
[u, v] = size(X);

if (n==v)
    sig = X * theta';
else if (l==v)
    sig = X * theta;
    else
    end
end

h= sigmoid(sig);
for i = 1:length(h)
if(h(i) >= 0.5)
    p(i) = 1;
else
    p(i) = 0;
end
    
end
