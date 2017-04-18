% Sourced from text "Pattern Recognition"

function z=bayesClassifier(m,S,P,X)
    [temp, c]=size(m);
    [temp, N]=size(X);
    for i=1:N
        for j=1:c
            val(j)=P(j)*gaussDensVal(m(:,j), S(:,:,j),X(:,i));
        end
        [temp,z(i)]=max(val);
    end
end