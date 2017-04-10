

%% Problem 2.7
clear all
clc

N = 1000;
m = [1 4 8; 1 4 1];
Si = [];
for i = 1:3
    Si(:,:,i) = [2 0; 0 2];
end
P1 = [1/3, 1/3, 1/3];
P2 = [0.8, 0.1, 0.1];

[X5, y1] = genGaussClasses(m, Si, P1, N);
%[X5_prime, y2] = genGaussClasses(m, Si, P2, N);

bayes_X5 = bayesClassifier(m, Si, P1, X5);
euclid_X5 = euclidClassifier(m, X5);

%bayes_X5_prime = bayesClassifier(m, Si, P2, X5_prime);
%euclid_X5_prime = euclidClassifier(m, X5_prime);

X5_bayes_error = computeError(bayes_X5, y1) 
%% Problem 2.8
clear all
clc




