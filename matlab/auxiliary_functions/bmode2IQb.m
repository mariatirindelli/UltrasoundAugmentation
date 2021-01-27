function IQb = bmode2IQb(bmode, DR, real_max)

I = double(bmode(:));
I = I .* DR / 255;

% Assuming IQb_max = 1
IQb = 10.^( (I - DR) ./ 20 )*real_max;

IQb = complex(IQb, 0);
% absolute mod cannot be reversed
end
