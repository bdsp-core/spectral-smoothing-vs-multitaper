function L = fcnFindLeakageSpSmMT2(f1,fp,findx,t,sig,df,tapers); 

% spectral smoothing using MT kernel -- W as a parameter; does not need to
% match W in MTSA being compared with this
s=sin(2*pi*t*f1);

%-------------------------
Ss = fcnSmSpect2(s',t,tapers,sig,findx);
ind = find(fp>=0); 
fidx = fp(ind); 
Ss = Ss(ind); 
Ss = Ss/sum(Ss); 
%-------------------------

%% measure leakage: % of power outside of f1+/-df
i0=find(fidx<=(f1+df) & fidx>=(f1-df)); 
i1=find(fidx>(f1+df) | fidx<(f1-df)); 
L=(sum(Ss)-sum(Ss(i0)))/sum(Ss)*100; 

