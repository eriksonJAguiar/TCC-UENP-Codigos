% $Author: Yasuko Matsubara 
% $Date: 2013-2-18

%--- input ------------------------------%
% T:   duration of the sequence
% params: model parameters 
%   params(1): N
%   params(2): betaN
%   params(3): delta
%   params(4): gamma
%   params(5): tv (starting time of vaccination)
%   params(6): Sv (strength of vaccination)
%   params(7): background noise
%   params(8): Pp (period)
%   params(9): Pa (strength of periodicity)
%   params(10): Ps (phase shift of periodicity)
%   
%   %%% optinonal (change start)
%   params(11): st    (starting point)
%   params(12): I(st) (starting count)
%
% ext: ext list: externals (ext: k x 3 (loc, wd, value))
%----------------------------------------%

%--- output ------------------------------%
% idx:	index (size: Tx1)
% S:    (size: Tx1)
% I:    (size: Tx1)
% V:    (size: Tx1)
%----------------------------------------%

%----------------------------------%
%% give the number of awake people at time-tick n
%----------------------------------%
function [S, I, V, P] = M_FunnelRE(T, params, ext)
    
    params=abs(params);
    %
    N       =  params(1);
    beta0   =  params(2)/N;
    delta   =  params(3);
    gamma   =  params(4);
    %
    tv      =  round(params(5))+1;
    nu0     =  params(6);
    bgn     =  params(7);
	%
    Pp      =  params(8);
    Pa      =  params(9);
    Ps      =  params(10);
    %
    S=(N)*ones(T,1); 
    I=(0)*ones(T,1); 
    V=(0)*ones(T,1); 
    P=(0)*ones(T,1); 
    C=(0)*ones(T,1); 
    
    % default
    if(length(params)==10)
        S(1) = N-1;
        I(1) = 1;
        V(1) = 0;
        st0=0; I0=0;
    elseif(length(params)==12)
        st0=round(params(11)+1);
        I0=params(12);
    end
     
    if(sum(isnan(params))>0); return; end

 
    %% --- periodic beta --- %% 
    for t=1 : T
    	P(t) = beta0*period(t, Pp, Pa, Ps);
    end
    %% ---  vaccination --- %% 
    if(isnan(tv)); tv=T; end
    for t=tv : T
    	C(t) = nu0;
    end
    %% ---  externals --- %% 
    E = external(T, ext);
	%% ---  main function --- %% 
    for t=1 : T-1      
        beta = P(t+1);
        nu   = C(t+1);
        e    = E(t+1);
        if(t==st0); bgn=I0; end
        %
        S(t+1) = S(t) - beta*S(t)*e*(I(t)+bgn) + gamma*V(t) - S(t)*nu;
        I(t+1) = I(t) + beta*S(t)*e*(I(t)+bgn) - delta*I(t);
        V(t+1) = V(t) + delta*I(t) - gamma*V(t) + S(t)*nu; 
        %
        if(abs(I(t+1)+S(t+1)+V(t+1)-N > 0.01) || I(t+1)<0) 
            S(t+1) = 0; I(t+1) = 0; V(t+1) = 0;
            %disp('error')
            break;
        end
    end
	%% ---  main function --- %% 
    % put outliers
    I = I + N*outliers(T, ext);
 

end

%----------------------------------%
%% outlier
%----------------------------------%
function [E] = outliers(T, ext)
    E=(0)*ones(T,1);
	if(isempty(ext)); return; end
    % if outlier
    ext=ext(ext(:,2)==-1, :);
    for i=1:size(ext,1)
        loc=ext(i,1);
        if(loc<1 || loc>T); continue; end
        E(loc) = ext(i,3);
    end
    E(E<0)=0;
    
end
%----------------------------------%
%% exo shock
%----------------------------------%
function [E] = external(T, ext)
    E=(1)*ones(T,1);
    if(isempty(ext)); return; end
    % if outlier, remove
    ext(ext(:,2)==-1, :)=[];
    loc = round(ext(:,1));
    wd  = round(ext(:,2));
    val = ext(:,3);
    for k=1:size(ext,1)
        st=loc(k)-wd(k);
        ed=loc(k)+wd(k);
        if(st<1);return;end
        if(ed>T);return;end
        E(st:ed)=E(st:ed)+val(k);
    end
end
%----------------------------------%
%% cyclic wave
%----------------------------------%
function [value] = period(t, Pp, Pa, Ps)
    if(Pa==0 || Pp <= 0)
        value=1;
    else
        value = 1 + Pa * cos((2*pi/Pp)*(t+Ps));
    end
end


