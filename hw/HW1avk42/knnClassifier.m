% Sourced from text "Pattern Recognition"

function z = knnClassifier(Z,v,k,X)
    [temp,N1]=size(Z);
    [temp,N]=size(X);
    c=max(v); 
    for i=1:N
        dist=sum((X(:,i)*ones(1,N1)-Z).^ 2);
        [sorted,nearest]=sort(dist);
        refe=zeros(1,c); 
        for q=1:k
            class=v(nearest(q));
            refe(class)=refe(class)+1;
        end
        [temp,z(i)]=max(refe);
    end
end