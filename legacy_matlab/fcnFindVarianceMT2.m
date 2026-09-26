function [V,Sm]=fcnFindVarianceMT2(f,fp,findx,t,W,K,s,tapers); 

%------------------------------------------------
Sm = fcnMTSA2(s,t,f,W,K,findx,tapers);
ind = find(fp>=0); 
fidx = fp(ind); 
Sm = Sm(ind); 
Sm = Sm/sum(Sm);  
%------------------------------------------------

V = var(Sm)*1e6; 
