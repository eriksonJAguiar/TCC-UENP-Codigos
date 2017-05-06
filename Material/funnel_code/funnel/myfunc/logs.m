function [y] = logs(x)
    y = 2.0*log2(x)+1.0;
    y(x==0)=0;
    
end