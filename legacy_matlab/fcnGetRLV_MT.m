function [dfm,Lm,Vm] = fcnGetRLV_MT(W,K,T,Fs,s,t,f1,f2,f,fp,findx); 

TW=T*W; 
Nt = length(s); 
tapers=[TW K]; 
tapers=dpsschk(tapers,Nt,Fs); % get tapers
dfm = fcnFindMTSAspecRes2(f,f1,f2,fp,findx,t,W,K,tapers);
Lm=fcnFindLeakageMT2(f,f1,fp,findx,t,W,K,tapers,dfm);
[Vm,Sm]=fcnFindVarianceMT2(f,fp,findx,t,W,K,s,tapers); 

