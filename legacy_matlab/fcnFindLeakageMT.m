function L=fcnFindLeakageMT(f1,findx,fp,t,f,W,K,df); 

% leakage for MTSA
s=sin(2*pi*t*f1);
[Sm,tapers] = fcnMTSA(s,t,f,W,K,findx); 

Sm=Sm(findx); 
ind = find(fp>=0); 
fidx = fp(ind); 
Sm = Sm(ind); 
Sm = Sm/sum(Sm); 
plot(fidx,Sm);

%% measure leakage: % of power outside of f1+/-df
i0=find(fidx<=(f1+df) & fidx>=(f1-df)); 
i1=find(fidx>(f1+df) | fidx<(f1-df)); 
L=(sum(Sm)-sum(Sm(i0)))/sum(Sm)*100; 

