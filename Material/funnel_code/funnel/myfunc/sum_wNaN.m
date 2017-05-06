function [ Y ] = sum_wNaN( X )
%MEAN_WNAN Summary of this function goes here
%   Detailed explanation goes here

[n, m]=size(X);

Y=zeros(1,m);
for i=1:m
    sum=0;
    cnt=0;
    for j=1:n
        if(isnan(X(j,i))==0)
            sum = sum+X(j,i);
            cnt = cnt+1;
        end
    end
    Y(i) = sum; %/cnt;
    
end

end

