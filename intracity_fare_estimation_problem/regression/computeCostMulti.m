function J = computeCostMulti(X, y, theta)
m = length(y); 
k = X*theta-y;
k=k.^2;
J=sum(k)/(2*m);
end
