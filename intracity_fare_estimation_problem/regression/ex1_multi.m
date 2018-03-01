clear ; close all; clc

data = load('sortvehicle.csv');
[m n]=size(data);
m=2798;
X = data(1:2798, 2:n-1);
y = data(1:2798, n);
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];
alpha = 1;
num_iters = 5000;

n=n-1;
theta =  randInit(1,n);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

price = 0; % You should change this
%data2=load('modified_test.csv');

price=X*theta;
[price(1:50) y(1:50)]