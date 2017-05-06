% $Author: Yasuko Matsubara 
% $Date: 2014-2-18

%--- input ------------------------------%
% dat:    input sequence 
% params: base parameters
% E0:     current externals (k x 3), if not, E=[]
% maxk:   max # of external-shocks&mistakes 
%----------------------------------------%

function [ RSE, E ] = M_fitExternal(dat, params, E0, maxk, wantPlot)
    TH=20; Pp = params(8);
	E=E0; 
	%% if data is too small, do nothing
	if(sum_wNaN(dat)<100 || max(dat)<TH); 
        disp('too small dat'); RSE=inf; return; end
    if(sum(dat>0)/Pp < maxk); maxk = floor(sum(dat>0)/Pp); end
    
    if(maxk>0) 
        %% if want to find optimal # of k (<maxk)
        E=[];
        %% incremental external fitting
        while (size(E,1)<maxk)
            %% find another external shock
            [RSE, ext] = M_fitExternal_single(dat, params, E);
            if(isempty(ext)); break; end
            C0=M_dist(dat, params, E, 'lin');
            C =M_dist(dat, params, [E; ext], 'lin');
            if(C0<C); ext(3) =0; end
          	E=[E; ext]; %#ok<AGROW>
        end
        
        %% ----- refine E list ----- %%
        if(~isempty(E))            
            % if empty, delete
            E(E(:,3)==0,:)=[];
            %% sort Externals
            if(~isempty(E))
                [tmp, idx] = sort(E(:,1)); E=E(idx,:);       
            end
        end
        %% ----- refine E list ----- %%

    
	elseif(maxk==-1)        
        %% if want to fit fixed # of k
        %%    (i.e., use fixed loc&wd, then, find optimal "ext.e")
        for i=1:size(E,1)
            ext=E(i,:); E(i,3) = 0;
            [err, ext]=search_e(dat, params, E, ext); 
            C0=M_dist(dat, params, E, 'C');
            C= M_dist(dat, params, [E; ext], 'C');
            %% avoid overfitting & null-data fitting
            if( abs(C0-C)<=0 );   ext(3) = 0; end
            % if missing values
            [tmp, st, ed] = M_splitSeq(dat);
            if( ext(1)<st || ext(1)>ed ); ext(3)=0; end

            E(i,:) = ext;
        end
    end  

    RSE=F_RNF(dat, params, E, [], []);
    
    %% --- if want plot --- %%
    if(wantPlot)
        figure(10); M_plotsRNF(dat, params, E, '')
    end
	%% --- if want plot --- %%

end


function [ RSE,  ext ] = M_fitExternal_single( dat, params, E )
    RSE = inf;
    T=length(dat);
    
	%% --- "external-shock" check --- %%
	% (a) find optimal location
    ext=init_ext0(0);  %(1);
	[err, ext] = search_loc(dat, params, E, ext);
	% (b) find optimal width & e
	[err, ext] = search_wd( dat, params, E, ext);
	%% --- "external-shock" check --- %%

    %% --- "mistake" check --- %%
    ext0=init_extMax(T, dat, params, E);
    ext2 = ext0; ext2(2)=-1;
    [err2, ext2]=search_e(dat, params, E, ext2); 
    if(err2<err); ext = ext2; end
    %% --- "mistake" check --- %%


end


%----------------------------------%
% ext setting
%----------------------------------%
%% initialize ext (reset value)
function [ext] = init_ext0(wd)
    %% model 
    ext(1)=0;
    ext(2)=wd;
    ext(3)=0.2;

end
%% initialize ext (find most likely outliers)
function [ext] = init_extMax(T, dat, params, E)
    %% model
    [S, I, R, P]=M_FunnelRE(T, params, E);
    
    if(1==0)
        %% --- remove current externals  
        if(~isempty(E)) 
            wds=params(8); %% i.e., Pp=52
            for i=1:size(E,1)
                loc=E(i,1); wd=E(i,2);
                wd=wd+wds;
                st=loc-wd; ed=loc+wd;
                if(st<1); st=1; end
                if(ed>T); ed=T; end
                I(st:ed)=0;
                dat(st:ed)=0;
            end
        end 
        %% --- remove current externals  
    end
    
    %I=sqrt(I+1); dat=sqrt(dat+1);
    diff=sqrt((I - dat).^2); %RSQRERR
    loc=find(diff==max(diff));
    if(isempty(loc)); ext=[]; return; end
    loc=loc(1); 
    ext(1)=loc;
    ext(2)=0;
    ext(3)=0.2;
    %% check nan values
    [tmp, st, ed] = M_splitSeq(dat);
    if(loc<st || loc>ed); ext=[]; end
end
%% constraint
function [ext] = const(ext)
    lb_e = -0.90;
    ub_e =  1000.0;
    if(ext(3)<lb_e); ext(3) = 0; end 
    if(ext(3)>ub_e); ext(3) = 0; end 
end

%----------------------------------%
% ext fitting
%----------------------------------%
%% search windowsize
function [err, ext]=search_e(dat, params, E, ext)   

	try
        ext(3) = lsqnonlin(@(x)F_RNF(dat, params, E, ext, x), ... 
              ext(3), [], [], M_setOpt());  
    catch
        ext(3) = 0;
	end
        ext = const(ext);

    err = F_RNF(dat, params, E, ext, ext(3));
  
end%% search windowsize
function [err, opt]=search_wd(dat, params, E, ext)
    wdlst = [0 1 2 4 8 16 32 64 128];
    elst  = [];
    opt=ext;
    sselist=inf(1,length(wdlst));
    for i=1:length(wdlst); 
        ext(2)=wdlst(i);
        [err,ext] = search_e(dat, params, E, ext);
        elst = [elst; ext]; %#ok<AGROW>
        sselist(i) = F_RNF(dat, params, E, ext, ext(3));
        if(i>1 && sselist(i-1)<sselist(i)); break; end
    end
    minlst=find(sselist==min(sselist));
    if(~isempty(minlst))
        opt = elst(minlst(1),:);
        err = min(sselist);
    end
end
%% search location
function [err, opt]=search_loc(dat, params, E, ext)
 
    [tmp, st, ed] = M_splitSeq(dat);
    Pp=52;
    %% first loop
    loclst=st:Pp:ed;
    sselist=zeros(1,length(loclst));
    for i=1:length(loclst);
        ext(1)=loclst(i);
        sselist(i) = F_RNF(dat, params, E, ext, ext(3));
    end
    minlst=find(sselist==min(sselist));
    loc=loclst(minlst(1));
    %% second loop
    loclst=ceil(loc-Pp):1:ceil(loc+Pp);
    sselist=zeros(1,length(loclst));
    for i=1:length(loclst);
        ext(1)=loclst(i);
        sselist(i) = F_RNF(dat, params, E, ext, ext(3));
    end
    minlst=find(sselist==min(sselist));
    if(~isempty(minlst))
        opt(1)= loclst(minlst(1));
     	err = min(sselist);
    end
end
function [err, opt]=search_loc_org(dat, params, E, ext)
    opt=ext;
    current=ext(1);
    wd=ext(2)+4;
    loclst=current-wd:1:current+wd;
    sselist=zeros(1,length(loclst));
    for i=1:length(loclst);
        ext(1)=loclst(i);
        sselist(i) = F_RNF(dat, params, E, ext, ext(3));
    end
    minlst=find(sselist==min(sselist));
    if(~isempty(minlst))
        opt(1)= loclst(minlst(1));
     	err = min(sselist);
    end

end
%% model distance function
function sse=F_RNF(dat, params, E, ext, x)
    if(~isempty(ext));ext(3)=x;end
    E=[E; ext];
    sse = M_dist(dat, params, E, 'lin'); 
    return; 
end



