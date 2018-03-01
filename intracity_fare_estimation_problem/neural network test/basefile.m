
clear ; close all; clc

ip  = 5;  
hide = 20;   
labels = 1;      
                        
data = load('train.csv');
[m n]=size(data);
x = data(:, 1:n-1);
x(:,n-1)= x(:,n-1).^2;
y = data(:, n);
[x mu sigma] = featureNormalize(x);
x = [ones(m, 1) x];

theta1=randInit(ip,hide);
theta2=randInit(hide,labels);
nn_params=[theta1(:) ; theta2(:) ];

lambda=1.5;


options = optimset('MaxIter', 200);
costFunction = @(p) nnCostFunction(p,ip,hide,labels,x,y,lambda);

[new_params,cost]=fmincg(costFunction,nn_params,options);
theta1=reshape(new_params(1:hide*(ip+1)),hide,ip+1);
theta2=reshape(new_params((1+hide*(ip+1)):end),labels,hide+1);

guess=predict(theta1,theta2,testx);  % see this too
[guess y]
%save submit.csv guess
