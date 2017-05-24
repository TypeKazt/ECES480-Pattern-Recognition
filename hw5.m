clear all
clc

%% 5.1

N1 = normrnd(0, 1, 100, 1);
N2 = normrnd(2, 1, 100, 1);
tt1 = ttest2(N1, N2);

if tt1 > 0
	fprintf('N1 = 100, Mean = 0, Sigma = 1\nN2 = 100, Mean = 2, Sigma = 1\n')
	fprintf('Significant difference\n\n')
end

N2 = normrnd(0.2, 1, 100, 1);
tt1 = ttest2(N1, N2);

if tt1 > 0
	fprintf('N1 = 100, Mean = 0, Sigma = 1\nN2 = 100, Mean = 0.2, Sigma = 1\n')
	fprintf('Significant difference\n\n')
end

N1 = normrnd(0, 1, 150, 1);
N2 = normrnd(2, 1, 200, 1);
tt1 = ttest2(N1, N2);

if tt1 > 0
	fprintf('N1 = 150, Mean = 0, Sigma = 1\nN2 = 200, Mean = 2, Sigma = 1\n')
	fprintf('Significant difference\n\n')
end


N2 = normrnd(.2, 1, 200, 1);
tt1 = ttest2(N1, N2);

if tt1 > 0
	fprintf('N1 = 150, Mean = 0, Sigma = 1\nN2 = 200, Mean = 0.2, Sigma = 1\n')
	fprintf('Significant difference\n\n')
end

%{
There was a significant difference in the each experiment with a mean delta of 2. 
The ttest attempts to determine the confidence that two data sets come from the 
same normal distribution with a mean of 0 and an unknown sigma. From the experiment 
it can be seen that data sets with a small difference in mean will be classified 
under the same normal distribution. This can relate to any learning model that uses
mean as a feature, and may classify data similarly. Mean alone would not be a 
good feature, but may provide insight as a support feature. 

Table 5.1 is calculated using a Gaussian Kernel.
%}

%% 5.2
clear all

I = [.2 0; 0 .2];
I(:,:,2) = I(:,:,1);
I(:,:,3) = I(:,:,1);
I(:,:,4) = I(:,:,1);

fprintf('Part a\n')
[N, classes] = genGaussClasses([-10 -10 10 10; -10 10 -10 10], I, [.25, .25, .25, .25], 400);
[Sw, Sb, Sm] = scatter_mat(N, classes)
j3 = J3_comp(Sw, Sm)
fprintf('\n')

fprintf('Part b\n')
[N, classes] = genGaussClasses([-1 -1 1 1; -1 1 -1 1], I, [.25, .25, .25, .25], 400);
[Sw, Sb, Sm] = scatter_mat(N, classes)
j3 = J3_comp(Sw, Sm)
fprintf('\n')

fprintf('Part c\n')
[N, classes] = genGaussClasses([-10 -10 10 10; -10 10 -10 10], I, [3, 3, 3, 3], 400);
[Sw, Sb, Sm] = scatter_mat(N, classes)
j3 = J3_comp(Sw, Sm)
fprintf('\n')


%% Spectrel Learning

%{
	1) The kernel applied is the Gaussian Kernel. The kernel is applied to the 
	higher dimensional data, which yields a similarity measure between each feature
	vector.

	2) The affinity matrix is a measure of similarity between data points, which
	is calculated by the Gaussian Kernel.

	3) The data that is clustered is the input data projected to a higher dimensionality. 
%}

