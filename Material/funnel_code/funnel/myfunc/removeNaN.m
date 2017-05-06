function [ x ] = M_removeNaN( x )
%M_REMOVENAN Summary of this function goes here
%   Detailed explanation goes here

%% -- interp -- %%
if(1==0)
	nanx    = isnan(x);
	t       = 1:numel(x);
	x(nanx) = interp1(t(~nanx), x(~nanx), t(nanx));
	x       = floor(x);
end

x(isnan(x))=0;
x(isinf(x))=0;
end

