% Sourced from text "Pattern Recognition"

function [Dv, classes] = genGaussClasses(m, S, P, N)
    [temp, c] = size(m);
    Dv = [];
    classes = [];
    for i = 1:c
        t = mvnrnd(m(:,i), S(:,:,i), fix(P(i)*N))';
        Dv = [Dv t];
        classes = [classes ones(1, fix(P(i)*N))*i];
    end
end