function [V,Sm]=fcnFindVarianceMT(findx,fp,t,f,W,K,s); 

% leakage for MTSA
[Sm,tapers] = fcnMTSA(s,t,f,W,K,findx); 
Sm=Sm(findx); 
ind = find(fp>=0); 
fidx = fp(ind); 
Sm = Sm(ind); 
Sm = Sm/sum(Sm); 
V = var(Sm)*1e6; 
