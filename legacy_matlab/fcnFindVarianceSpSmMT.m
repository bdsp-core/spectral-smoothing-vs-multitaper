function [V,Ss] = fcnFindVarianceSpSmMT(fp,findx,s,t,sig,tapers) 

%-------------------------

Ss = fcnSmSpect2(s',t,tapers,sig,findx);
ind = find(fp>=0); 
fidx = fp(ind); 
Ss = Ss(ind); 
Ss = Ss/sum(Ss); 
%-------------------------
  
V = var(Ss)*1e6; 