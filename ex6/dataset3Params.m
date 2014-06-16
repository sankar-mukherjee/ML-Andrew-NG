function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 10;
%sigma = 0.3;
A = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
B=[];
CC=[];
S=[];
m=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for i=1:length(A)
    C=A(i);
    for j=1:length(A)
        sigma = A(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        m=mean(double(predictions ~= yval));
        B = [B;m];       
        CC =[CC; C];
        S =[S; sigma];
    end
end

[val index]=min(B);
C= CC(index);
sigma=S(index);
%  c=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]';
% sigma=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% cr=repmat(c,8,1);
%  crr=cr(:);
%  sigmar=repmat(sigma,8,1);
%  para=[crr sigmar];
%  error=[];
%  for i=1:length(para)
%      model= svmTrain(X, y, para(i,1), @(x1, x2) gaussianKernel(x1, x2, para(i,2)));
%      predictions = svmPredict(model, Xval);
%      error= [error mean(double(predictions ~= yval))]+0*i;
%  end
% 
% [dummy dim]=min(error');
% C=para(dim,1);
% sigma=para(dim,2);
% =========================================================================

end
