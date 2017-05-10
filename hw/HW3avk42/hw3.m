
%% Problem 4.1
clear all
clc

w_1 = [.1 -.2;
	   .2 .1;
	   -.15 .2;
	   1.1 0.8;
	   1.2 1.1];

w_2 = [1.1 -.1;
	   1.25 .15;
	   .9 .1;
	   .1 1.2;
	   .2 .9];


scatter(w_1(:,1), w_1(:, 2))
hold on
xlim([-2.5 2.5])
ylim([-2.5 2.5])
title('Problem 4.1')
scatter(w_2(:,1), w_2(:, 2))
legend('w1', 'w2')

% It is clear that the data set is not linearly separable 
% by one line, but it can be separated by two. This is where
% a multilayer perceptron will be useful.

%% Deep Learning

% a)

% A convolutional neural net is strictly feed forward where the combination of
% multiple functions at neuron yield a new function (f + a = g), hence the included term "convolution".
% A neural net is generic term for a set of machine learning algorithms that utilize "neurons".

% b)

% A feed forward neural net only has data flowing in one direction, usually towards the output neuron. 
% From a graph perspective, there are no cycles in the graph for forward propagation. 
% A neural net with back propagation is implemented similarly to a feed forward, except error
% is calculated at each layer and fed back to previous layers to retrain the network. 


%% Clustering by compression

% a)

% I'm very sleepy


