function [X] = M_removeSparse(X, wd)
    wd=ceil(wd/2);
    n=length(X);
    for t=1:n
        st=t-wd;ed=t+wd;
        if(st<1); st=1;end;
        if(ed>n); ed=n;end;        
        counts=sum(isnan(X(st:ed)));
        len=ed-st;
        %if(counts>wd) 
        if(counts>len/2) 
            X(t)=NaN;
        end
    end
end