%% for visualization (LOG & LIN scale)
function [RMSE] = M_plotsRNF(dat, D, E, outfn)

	RMSE=printRNF(dat, D, E, 1);
    if(sum(isnan(D))>0); return; end

    T=length(dat);
    [S, I, V, P]=M_FunnelRE(T, D, E);
    idx=1:T;
    
    [daty, st, ed] = M_splitSeq(dat);
    subplot(2,1,1)
    %% linear plot
    plot(idx, dat, 'o', 'color',[0.6, 0.6, 0.6]);
    hold on
    plot(idx, I, 'r-');
    legend('Original','I(t)');
    title([ ...
        'Er= ',       num2str(RMSE,      '%.2e'), ...
        ' (' ,       num2str(D(1), '%.1e'), ...
        ', ',      num2str(D(2), '%.2f'), ...
        ', ',     num2str(D(3), '%.2f'), ...
        ', ',     num2str(D(4), '%.2f'), ...
        ') , P=(',        num2str(D(9), '%.1f'), ...
        ', ',        num2str(D(10), '%.0f'), ...
        '), (',        num2str(D(5), '%.0f'), ...
        ', ',        num2str(D(6), '%.2f'), ...
        '), k=', num2str(size(E,1), '%.0f')]);   
    xlim([st,ed])
    xlabel('Time (t)');
    ylabel('Count');
    hold off

    subplot(2,1,2)
    %% semi-log plot
    I=(I+1);
    S=(S+1);
    dat=(dat+1);
    semilogy(idx, dat, 'o', 'color',[0.6, 0.6, 0.6])
    hold on
    semilogy(idx, S, 'b--'); 
	semilogy(idx, I, 'r-'); 
    semilogy(idx, V, '--','color',[0.1, 0.5, 0.1]); 
    legend('Original', 'S(t)', 'I(t)', 'V(t)');
    
  	xlim([st,ed])
    ylim([1, max(S)*10])
    xlabel('Time (t)');
    ylabel('Count (log)');
    hold off
    if(~isempty(outfn))
          disp(['save as: ', [outfn,'']])
          saveas(gcf, [outfn,''], 'fig');
          saveas(gcf, [outfn,''], 'png');
    end
end

%% for visualization 
function [RSE_LIN] = printRNF(dat, params, E, fid)
    RSE_LIN = M_dist(dat, params, E, 'lin');
    %% output parameters
    fprintf(fid, '===================================\n');
    fprintf(fid, ['N       = ',  num2str(params(1), '%.0f'), '\n']);
    fprintf(fid, ['betaN   = ',  num2str(params(2), '%.2f'), '\n']);
    fprintf(fid, ['delta   = ',  num2str(params(3), '%.2f'), '\n']);
    fprintf(fid, ['gamma   = ',  num2str(params(4), '%.2f'), '\n']);
    fprintf(fid, '-----------------------------------\n');
    fprintf(fid, ['pcycle (Pp, Pa, Ps) = ', ...
                                num2str(params(8)), ' ', ...
                                num2str(params(9)), ' ', ...
                                num2str(params(10)), '\n']);
    fprintf(fid, '-----------------------------------\n');
    fprintf(fid, ['thetaT      = ',  num2str(params(5), '%.0f'), '\n']);
    fprintf(fid, ['theta0      = ',  num2str(params(6), '%.3f'), '\n']);
    fprintf(fid, '-----------------------------------\n');
    fprintf(fid, ['error (LIN)  = ',   num2str(RSE_LIN, '%.2f'), '\n']);        
    fprintf(fid, '===================================\n');
end
