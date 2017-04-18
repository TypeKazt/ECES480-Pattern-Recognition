% Sourced from text "Pattern Recognition"

function z=euclidClassifier(m, X)
    [temp,c]=size(m); 
    [temp,N]=size(X);
    for i=1:N
        for j=1:c
           val(j)=sqrt((X(:,i)-m(:,j))'*(X(:,i)-m(:,j)));
        end
        [temp,z(i)]=min(val);
    end
end