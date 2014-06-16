function [jVal, gradient] = costFunction(theta)
%COSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here
jVal = (theta(1)-5)^2 + (theta(2)-5)^2;

gradient = zeros(2,1);
gradient(1)=2*(theta(1)-5);
gradient(2)=2*(theta(2)-5);
end

