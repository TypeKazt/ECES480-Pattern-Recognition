% Sourced from text "Pattern Recognition"

function val = gaussDensVal(m, S, x)
    [l, q] = size(m);
    val = (1/((2*pi)^(l/2)*det(S)^0.5))*exp(-0.5*(x-m)'*inv(S)*(x-m));
end