% $Author: Yasuko Matsubara 
% $Date: February, 2014

addpath('./myfunc/')

%% demo: fitting global-level sequence
%label='17_measles';
label='37_typhoidfever'; 
%label='Base_dengue_CA_data';  
%label = '12_influenza';

    outdir='./_out/';
    mkdir(outdir)
    fn=['./_dat/',label];
    outfn=[outdir,label];
    outfnL=[outfn,'_Fl'];
   
    % load data
    Xorg=load(fn);

    %% --------------------------- %%
    %% only global fitting
    [BR_global, Extra_global, N_local, Extra_local] = M_FunnelFit(Xorg, outfn, [], 0);
    %% if global&local fitting
    %%[BR_global, Extra_global, N_local, Extra_local] = M_FunnelFit(Xorg, outfn, outfnL, 0);
    %% --------------------------- %%

    %% Funnel parameters ...
    % BR_global:    global parameters (B & R)
    % Extra_global: global parameters (E & M)
    % N_local:      local  parameters (N)
    % Extra_local:  local  parameters (E & M) 
