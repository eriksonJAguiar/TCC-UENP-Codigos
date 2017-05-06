% $Author: Yasuko Matsubara 
% $Date: 2013-10-25

%--- input ------------------------------%
% dat:   input sequence 
% outfn: output file name
% iter:  # of max iteration
%params0: initial params, if required; otherwise, params0=[];
%----------------------------------------%

function [RSE, params] = M_fitBasic( dat, Pp, outfn, iter, params0, E, fixlst, wantPlot)

    %% if data size is too small, do nothing
	if(sum_wNaN(dat)<100 || sum(dat>0)<Pp || max(dat)<10 )
        disp('too small dat')
        nparam=10;
    	params = NaN*zeros(1,nparam); 
        RSE=inf;
        if(~isempty(outfn)) 
            save([outfn,'.param'], 'params', '-ascii');
        end
        return;
	end
    disp(outfn)
    %% fitting model
    [RSE, params] = LMFit(dat, Pp, iter, params0, E, fixlst, wantPlot);
    if(isempty(outfn))
        return;
    end
    %% plot fitting results 
    M_plotsRNF(dat, params, E, outfn);
    %% save parameters
    save([outfn,'.param'], 'params', '-ascii');
    
end


function [RSE, params] = LMFit( dat, Pp, iter, params0, E, fixlst, wantPlot )

    %% --- parameter settings ---%%
    % duration of sequence
    T=length(dat);
    % if dat has a small tail-part
    hasaTail=(sum(dat<100)>sum(dat>0)*0.1);% && (sum(dat>100)>sum(dat>0)*0.1);
    % init parameters
    if(isempty(params0))
        [params]=init_params(dat, Pp);
    else
        params = params0;
    end
    [LBs, UBs]=init_const(dat, Pp, T);
    % params settings
    options=M_setOpt(); 
    % fitting order
    order=[];
    order=[order, 2 1 3 4]; % N, betaN, delta, gamma   
    order=[order, 10 9]; % pshift, prate
	order=[order, 7]; % background noise 
	order=[order, 5 6]; % tc Sc
    if(~isempty(fixlst))
        for i=1:length(fixlst)
            if(fixlst(i)==1)
                order(order==i)=[];
            end
        end
    end
    %% --- parameter settings ---%%

    %% start fitting
    RSEP=inf;
    MAXITER=iter; 
    for i=1: MAXITER
        if(wantPlot)  %% if you want to plot 
            figure(10)
            M_plotsRNF(dat, params, E, []);
            disp(['iter=', num2str(i), '/', num2str(MAXITER)]); 
        end
        for loc=1: length(order)
            lo=order(loc);
            
            if(lo==5) % tv: starting point of vaccination
                [params(lo)] = FD_search(dat, params, E, lo, 'lin');
            else      % others
                try
                    [params(lo)] = lsqnonlin(...
                    	@(x)F_RNF(dat, params, E, lo, x, 'lin'), ... 
                        	params(lo), [], [], options);  
                catch
                   	if(isempty(params0))
                        [params]=init_params(dat, Pp);
                    else
                        params = params0;
                    end
                end
                params = const(params, LBs,UBs);
            end
        end
        % compute RSError
        Delta = 1.0; %0.1;
        %RSE = printRNF(dat, params, E, 1);
        RSE = F_RNF(dat, params, E, -1, -1, 'lin');
        if( i>3 && (abs(RSEP-RSE)<Delta) )
            break; %% final fitting
        end
        RSEP = RSE;
    end
    
    %% if has a tail part, then, fit in log scale
    disp('small part fitting...');
	if(hasaTail);
        lo=6; valorg=params(lo);
        try
            [params(lo)] = lsqnonlin(...
                @(x)F_RNF(dat, params, E, lo, x, 'log'), ... 
                params(lo), [], [], options);  
        catch
                params(lo) = valorg;
     	end
	end

   

end

function [params] = init_params(dat, Pp)
    %% init params
    % RNF-base
    N = 5*max(dat); 
    betaN = 1.0; 
    delta = 0.5;
    gamma = 0.1; 
    % RNF-X
    tC=length(dat);
    C0=0.000;
    bgnoise=0.0; 
	% RNF-P
    Pa=0.1;
    Ps=0;
    %I0=max(dat)/2;
    %R0=10;   %max(dat);
    params=zeros(1,9);
    params(1) = N;
    params(2) = betaN;
    params(3) = delta;
    params(4) = gamma;
    params(5) = tC;
    params(6) = C0;
    params(7) = bgnoise;
 	  params(8) = Pp;
    params(9) = Pa;
    params(10) = Ps;
    %params(11) = I0;
    %params(12) = R0;
end

%%
function [LBs, UBs]=init_const(dat, Pp, T)
    LB_base   = [max(dat)      0.5     1/4    0];  %% N, betaN, delta, gamma
    UB_base   = [10*max(dat)   2.0     1/1    1];  
    LB_X      = [0         0.00       0.0];         %% tC, C0, bgn
    UB_X      = [T         0.1       0.0];
	LB_P      = [Pp        0.001      0.0];         %% Pp, Pa, Ps
    UB_P      = [Pp        0.5        Pp];
    %%
    LBs       = [LB_base,       LB_X        LB_P    ];      
    UBs       = [UB_base,       UB_X        UB_P    ];
end

%%
function [params] = const(params, LB, UB)
    params = abs(params);
   	%% Ps = mod(Ps,Pp)
    params(10) = mod(params(10), params(8));
    %% L & U bounding
    params(params < LB) = LB(params < LB);
    params(params > UB) = UB(params > UB);
end

% discrete fitting
function [estimate] = FD_search(dat, params, E, loc, scale)
    st=1; ed=length(dat);
    Pp=params(8);
    idxlist=st:Pp:ed;
    sselist=zeros(1,length(idxlist));
    for i=1: length(idxlist);
        params(loc) = idxlist(i);
        sselist(i)=F_RNF(dat, params, E, -1, -1, scale);
    end
    minlst=find(sselist==min(sselist));
    if(~isempty(minlst))
        estimate= idxlist(minlst(1));
    else
        estimate=1;
    end
end
%----------------------------------%
% Rise and Fall fitting
%----------------------------------%
function sse=F_RNF(dat, params, E, loc, x, scale)
    if(loc~=-1); params(loc)=x; end
    sse = M_dist(dat, params, E, scale);
    return;  
end


