% Rerturns ratio of incorrect classification in a linear classifier

function r=verifyVector(X, y, w)
	n = 0;
	[l, N] = size(y);
	
	for i=1:N
		if w'*X(:,i)*y(i) < 0
			n = n+1;
		end
	end

	r = n/N;
end
