function z=euclidClassifier(m, X)
    [l,c]=size(m); % l=dimensionality, c=no. of classes
    [l,N]=size(X); % N=no. of vectors
    for i=1:N
        for j=1:c
           t(j)=sqrt((X(:,i)-m(:,j))'*(X(:,i)-m(:,j)));
        end
        % Determining the maximum quantity Pi*p(x|wi)
        [num,z(i)]=min(t);
    end
end