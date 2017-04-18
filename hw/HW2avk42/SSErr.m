% Sourced from "Pattern Recognition"

function w=SSErr(X,y)
	w=inv(X*X')*(X*y');
end
