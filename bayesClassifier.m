function z=bayesClassifier(m,S,P,X)
    [l,c]=size(m); % l=dimensionality, c=no. of classes
    [l,N]=size(X); % N=no. of vectors
    for i=1:N
        for j=1:c
            t(j)=P(j)*gaussDensVal(m(:,j), S(:,:,j),X(:,i));
        end
        % Determining the maximum quantity Pi*p(x|wi)
        [num,z(i)]=max(t);
    end
end