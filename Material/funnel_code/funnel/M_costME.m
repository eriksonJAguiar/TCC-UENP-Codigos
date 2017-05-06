function [ costM ] = M_costME(k)
%COSTME Summary of this function goes here
%   Detailed explanation goes here

d=40;
n=5252;
fB=M_fB();
costM = logs(k)  +  k*(log2(d)+ 2*log2(n) + fB);

end


function [ costM ] = M_costME_ex(Elst)
%COSTME Summary of this function goes here
%   Detailed explanation goes here
nE=length(Elst);
costM=[];
d=40;
n=5252;
fB=M_fB();

for i=1:nE
    costM(i) = 0;    %#ok<AGROW>
    E=Elst{i}; 
    %% if null, cost=0;
    if(size(E,1)==0); continue; end
    %% remove zero values
    E(E(:,3)==0,:)=[];
    outlier=E(E(:,2)==-1,:);
    E(E(:,2)==-1,:)=[];
    costO = size(outlier,1)*(log2(d)+log2(n) + fB);% + %sum(logs(outlier(:,3)));
    %costE = size(E,1)*(log2(d)+2*log2(n) + fB);
    costE = size(E,1)*(log2(d) + log2(n) + fB) + sum(logs(E(:,2)));
    kO = size(E,1);
    kE = size(outlier,1);
    costM(i) = costO + costE + logs(kO) + logs(kE); %#ok<AGROW>
end

costM=costM';

end
