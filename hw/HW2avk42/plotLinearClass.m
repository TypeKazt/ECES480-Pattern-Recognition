% Sourced from text "Pattern Recognition"

function plotLinearClass(X,y,m,w,name)
	plotData(X,y,m,name);
	x = [-10, 10];
	y = (w(3) - w(1)*x)/w(2);
	plot(x,y)
	hold off
    
end
