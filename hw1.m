
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
% Data plot of X5 with known classes
plotData(X5, y1, m, 'Data plot of X5 with known classes');
[X5_prime, y2] = genGaussClasses(m, Si, P2, N);
plotData(X5_prime, y2, m, 'Data plot of X5 prime with known classes');

bayes_X5 = bayesClassifier(m, Si, P1, X5);
plotData(X5, bayes_X5, m, 'Data plot of X5 Bayesian classifier');

euclid_X5 = euclidClassifier(m, X5);
plotData(X5, euclid_X5, m, 'Data plot of X5 Euclid classifier');

bayes_X5_prime = bayesClassifier(m, Si, P2, X5_prime);
plotData(X5_prime, bayes_X5_prime, m,...
    'Data plot of X5 prime Bayesian classifier');

euclid_X5_prime = euclidClassifier(m, X5_prime);
plotData(X5_prime, euclid_X5_prime, m, ...
    'Data plot of X5 prime Euclid classifier');

X5_bayes_error = computeError(bayes_X5, y1) 
X5_euclid_error = computeError(euclid_X5, y1) 

X5_prime_bayes_error = computeError(bayes_X5_prime, y2) 
X5_prime_euclid_error = computeError(euclid_X5_prime, y2) 


%% Problem 2.8

clear all
clc

N = 1000;
m = [1 8 13; 1 6 1];
Si = [];
for i = 1:3
    Si(:,:,i) = [6 0; 0 6];
end
P1 = [1/3, 1/3, 1/3];

% No requirementes on data generated, so for ease Gauss is used
[Z, y1] = genGaussClasses(m, Si, P1, N);
[X3, temp] = genGaussClasses(m, Si, P1, N);

knn1_X3 = knnClassifier(Z, y1, 1, X3);
plotData(X3, knn1_X3, m, 'Data plot of X3 with KNN classifier k=1');
knn11_X3 = knnClassifier(Z, y1, 11, X3);
plotData(X3, knn11_X3, m, 'Data plot of X3 with KNN classifier k=11');



