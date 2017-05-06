function [ Y, st, ed ] = M_splitSeq( X )
%M_SPLITSEQ Summary of this function goes here
%   Detailed explanation goes here
X=removeNaN(X);
[n, m]=size(X);
st=-1;
ed=-1;
for t=1:n
    if(st<0 && sum(X(t,:))>0)
        st=t;
        break;
    end
end
for t=n:-1:1
    if(ed<0 && sum(X(t,:))>0)
        ed=t;
        break;
    end
end
if(st==ed) ed=st+1; end
if(st==-1 || st<1)st=1;end
if(ed==-1 || ed>n)ed=n;end

Y=X(st:ed,:);
end

