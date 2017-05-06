%% model distance function
function [val] = M_dist(dat, D, E, scale)
    if(strcmp(scale,'C')==1)
        %val = cost(dat, D, E, 'INT');
        val = cost(dat, D, E, 'PDF');
    else
        val = distance(dat, D, E, scale);
    end
end

function [C] = cost(dat, D, E, scale)
    C=0;
    %% costC
        T=length(dat);
        [S, I, V, P]=M_FunnelRE(T, D, E);
        if(isnan(sum(I)) || sum(I)==0);
            C=inf; return;
        end   
        
        if(strcmp(scale,'INT')==1)
            val = (sum_wNaN(logs( ceil((I - dat).^2)) ));  
            C=val;
        elseif(strcmp(scale,'PDF')==1)
            %I=log(I+1); dat=log(dat+1);
            A=(I-dat).^1; %2;
            C=nansum(-log2(normpdf(A, 0, nanvar(A))));
        end
end
function [sse] = distance(dat, D, E, scale)
    T=length(dat);
    [S, I, V, P]=M_FunnelRE(T, D, E);
    if(isnan(sum(I))) 
        sse=inf;
        return;
    end

    if(strcmp(scale,'NRMSE')==1)
        % normalized root mean square error
        I   = I./sum_wNaN(I);
        dat = dat./sum_wNaN(dat);
    elseif(strcmp(scale,'lin')==1)
        I=I;
        dat=dat;
    elseif(strcmp(scale,'log')==1)
        I=log(I+1);
        dat=log(dat+1);
    elseif(strcmp(scale,'R2')==1)
        I=I.^(1/2);
        dat=dat.^(1/2);
    elseif(strcmp(scale,'R5')==1)
        I=I.^(1/5);
        dat=dat.^(1/5);
    elseif(strcmp(scale,'rate')==1)
        I = dat./I;
        I(isnan(I))=1;
        I(I>100)=1; % to avoid over fitting
        dat=ones(length(I),1);
    else
        error('use: C/lin/log/R2/R5/rate')
    end
    sse = sqrt(mean_wNaN((I - dat).^2));    
end
