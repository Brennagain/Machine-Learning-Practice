function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
ts = size(theta);
grad = zeros(ts);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
expression = -y.*log(h) - (1-y).*(log(1-h));
Jnonreg = (1/m)*sum(expression);
reg = (lambda/(2*m))*(theta(2:ts)'*theta(2:ts));
J = Jnonreg + reg;

gradexpression(1) = (1/m)*(h - y)'*X(:,1);
for j = 2:size(X,2)
gradexpression(j) = (1/m)*(h - y)'*X(:,j) + (lambda/m)*theta(j);
end

grad = gradexpression';




% =============================================================

end
