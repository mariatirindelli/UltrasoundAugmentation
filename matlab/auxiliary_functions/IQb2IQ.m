function IQ = IQb2IQ(IQb, Mdas)
% IQb = Mdas*IQ(:); --> Mdas*IQ(:) = IQb
% A = Mdas
% y = IQ(:)
% B = IQb
% y = A\B
B = IQb(:);
dA = decomposition(Mdas, "cod"); % the call dA\b returns the same vector as A\
IQ = dA\B;
end