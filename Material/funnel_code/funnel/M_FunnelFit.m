% $Author: Yasuko Matsubara 
% $Date: 2014-2-18
function [BR_global, Extra_global, N_local, Extra_local] = M_FunnelFit(Xorg, outfn, outfnL, SCALABILITY)
BR_global=[]; Extra_global=[]; N_local=[]; Extra_local=[];
%% --- setting --- %%    
    outfig=outfn;
    wantPlot=1;
    wantGmap=0; %% Gmap plot, default:NO
    if(SCALABILITY); outfig=[]; wantPlot=0; end
    % period (1 year = 52 weeks)
    Pp=52;
    % # of iteration
    ITER=10;
    

%% --- fitting --- %%    
    % split NaNs (tail part)
    [Xorg_s, st, ed] = M_splitSeq(Xorg); Xorg=Xorg(1:ed,:);
    % compute marginal value
    X=sum_wNaN(Xorg'); X=X'; X(X==0)=NaN;    
    X=M_removeSparse(X,Pp);
    
    %% ---------------------- %%
    %%     global fitting     %%
    %% ---------------------- %%
    [BR_global, Extra_global] = GlobalFit(X, Pp, ITER, outfn, outfig, wantPlot);
    %% ---------------------- %%
    %%     local fitting      %%
    %% ---------------------- %%
    if(~isempty(outfnL)) 
	mkdir([outfnL,'/'])    
    [N_local, Extra_local] = LocalFit(Xorg, Pp, ITER, outfn, outfnL, wantPlot, wantGmap, SCALABILITY);
    end
%% --- fitting --- %%

end


%% ---------------------- %%
%%     global fitting     %%
%% ---------------------- %%
% BR_global    : disease-global, i.e., B,R matrices
% Extra_global : extra-global,   i.e., E^(D/T) & M^(D/T) matrices
function [BR_global, Extra_global] = GlobalFit(X, Pp, ITER, outfn, outfig, wantPlot)

        BR_global = []; Extra_global=[1 1 1 1]; maxk=5; Clst=[];
        for kp=1:10 %try several numbers of "k=k^{kp}"
            %% base fitting
            [RSE, BR_global]=M_fitBasic(X, Pp, outfig, ITER, BR_global, Extra_global, [], wantPlot);
            if(isinf(RSE)); break; end
            %% check costC 
            C=M_dist(X, BR_global, Extra_global, 'C'); 
            k = size(Extra_global,1);
            Clst = [Clst; C  k C+M_costME(k)]; %#ok<AGROW>
            %% -----------------------------------%%
            if(wantPlot)
                %% --- plot costC & costT --- %%
                figure(7);clf; hold on;
                plot(Clst(:,2), Clst(:,3), 'r*-')
                plot(Clst(:,2), Clst(:,1), '*-'); hold off;
                xlabel('# of extras');
                ylabel('Cost')
                legend('CostT', 'CostC');
                saveas(gcf, [outfn, 'Clst'],'png');
                saveas(gcf, [outfn, 'Clst'],'fig');
                %% --- plot costC & costT --- %%
            end
            %% -----------------------------------%%
            Extra_local{kp} = Extra_global;   %#ok<AGROW>
            if(kp>2 && Clst(kp-1,2)>= Clst(kp,2)); break; end 
            %if(Clst(kp,2)>= 10 && Clst(kp-1,3)<Clst(kp,3)); break; end 
            if(Clst(kp,2)>= 4 && Clst(kp-1,3)<Clst(kp,3)); break; end 
            %% external fitting
            [RSE, Extra_global]=M_fitExternal(X, BR_global, [], maxk, wantPlot);
            maxk=maxk*2;
        end
        %% === (BEGIN) find optimal k with CostC & costM === %%
        if(sum(isnan(BR_global))==0)
            crrlst= Clst(:,3); 
            minlst=find(crrlst==min(crrlst));
            Extra_global = Extra_local{minlst(1)};
            %% final fitting (base)
            [RSE, BR_global]=M_fitBasic(X, Pp, outfig, ITER*2, BR_global, Extra_global, [], wantPlot);
            [RSE, Extra_global]=M_fitExternal(X, BR_global, Extra_global, -1, wantPlot);
            M_plotsRNF(X, BR_global, Extra_global, outfig)

        end
        %% === (END) find optimal k with CostC & costM === %%
        save([outfn, '_D'], 'BR_global', '-ascii')
        save([outfn, '_E'], 'Extra_global', '-ascii')       
end


%% ---------------------- %%
%%     local fitting     %%
%% ---------------------- %%
function [N_local, Extra_local] = LocalFit(Xorg, Pp, ITER, outfn, outfnL, wantPlot, wantGmap, SCALABILITY)
        m=size(Xorg, 2);
        fixlst=ones(1,12);
        fixlst(1)=0; % estimate N, only
        Dlst=[]; Extra_local=[];
        BR_global = load([outfn, '_D']);
        Extra_global = load([outfn, '_E']);
        for STATE=1:m 
            outfns=[outfnL,'/s',num2str(STATE,'%02d')];
            %% -----------------------------------%%
            if(SCALABILITY); outfns=[]; end
            %% -----------------------------------%%
            X=Xorg(:,STATE); X(X==0)=NaN;
            X=M_removeSparse(X,Pp);
            %
            E_local=Extra_global;
            D_local=BR_global;
            % reset externals
            E_local(:,3) = 0;
            %% --- inference --- %%
            for iter=1:2 %ITER
                [RSE, D_local]=M_fitBasic(X, Pp, outfns, ITER, D_local, E_local, fixlst, wantPlot);
                [RSE, E_local]=M_fitExternal(X, D_local, E_local, -1, wantPlot);
            end
            M_plotsRNF(X, D_local, E_local, outfns)
            %% --- inference --- %%
            
            % D locals
            Dlst=[Dlst; D_local]; %#ok<AGROW>
            % E locals
            Em=E_local(:,3);
            Extra_local = [Extra_local; Em']; %#ok<AGROW>
        end
                     
        N_local = Dlst(:,1);
        save([outfn, '_N'], 'N_local', '-ascii')
        save([outfn, '_Em'], 'Extra_local'  , '-ascii')
        
        
        %% ---  if plot Gmap ------------------%%
        if(~SCALABILITY && wantGmap)
            M_plot_Gmap(N_local, outfn);
            %% --- create Gmap (externals) --- %%
            Extra_local = load([outfn, '_Em']);
            %N_local = load([outfn, '_N']);
            Extra_local(Extra_local<0)=0;
            [tmp, k] = size(Extra_local);
            mkdir([outfn,'_Emap/'])
            for i=1:k
                e=Extra_local(:,i).*100; %N_local; %1000;
                M_plot_Gmap(e, [outfn,'_Emap/k', num2str(i)])
            end
            %% ------------------------------- %%
        end
        %% -----------------------------------%%
end
