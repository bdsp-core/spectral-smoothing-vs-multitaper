function L=fcnFindLeakageMT2(f,f1,fp,findx,t,W,K,tapers,df); 

% leakage for MTSA
s=sin(2*pi*t*f1);

%------------------------------------------------
Sm = fcnMTSA2(s,t,f,W,K,findx,tapers);
ind = find(fp>=0); 
fidx = fp(ind); 
Sm = Sm(ind); 
Sm = Sm/sum(Sm);  
%------------------------------------------------


%% measure leakage: % of power outside of f1+/-df
i0=find(fidx<=(f1+df) & fidx>=(f1-df)); 
i1=find(fidx>(f1+df) | fidx<(f1-df)); 
L=(sum(Sm)-sum(Sm(i0)))/sum(Sm)*100; 

