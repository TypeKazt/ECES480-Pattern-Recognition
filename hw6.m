clear all
clc

%% 6.1

l = 2;
N = 1000;
w = [1;1];
w0 = 0;
a = 10;
e = 1;
sed = 0;


% Generate data points
X = generate_hyper(w, w0, a, e, N, sed);

% Generate covariance matrix, eigenvectors, and variance
[pc, variances] = pcacov(cov(X'))

% orthogonal to w
h = [-1; 1];

%{
	The vector orthogonal to 'w' is [-1; 1], which points in the negative
	'X' direction, and positive 'Y' direction. The first principal component 
	has the same direction, but at a different magnitude (~.7). 
%}

%% MDL

%{
	1) MDL, BIC, and AIC are all a form of model selection for a given data set. 
	The difference between the three is their criteria for best model. MDL focuses
	on maximum compression, BIC uses the likelihood function to determine the criteria, 
	and AIC uses relative likelihood from other models to find the best model. 

	2) Using MDL for 6.1 may be possible by using a set of compression models similar
	to SCA, and allowing MDL to find the appropriate model. 
	
	3) Attempting to form the data in such a way that is both expressible and compressible. 
%}

