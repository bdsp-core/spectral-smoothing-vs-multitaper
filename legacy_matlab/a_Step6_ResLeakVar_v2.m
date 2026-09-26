clear all; clc; format compact; 

%% cleaner version of res/leak/var calculation; add in special spectrum later

%% try papoulis window
%% try special spectrum

%% define signal
T = 2; % Length of timeseries
Fs = 200; % Sampling rate 
signalType = 0; 

% problem -- signal does not have desired characteristics; ignores T
[s,t,f,f1,f2,fp,findx,Sxx] = fcnGetSignal(T,Fs,signalType);

%% get spectral resolution -- MTSA
W=1.5; 
TW=T*W; Nt = length(s); K=floor(2*TW-1); tapers=[TW K]; tapers=dpsschk(tapers,Nt,Fs); % get tapers
dfm = fcnFindMTSAspecRes2(f,f1,f2,fp,findx,t,W,K,tapers);
Lm=fcnFindLeakageMT2(f,f1,fp,findx,t,W,K,tapers,dfm);
[Vm,Sm]=fcnFindVarianceMT2(f,fp,findx,t,W,K,s,tapers); 
disp([dfm Lm Vm]);


%% get spectral resolution -- smoothing
W = .5; 
TW=T*W; Nt = length(s); K=floor(2*TW-1); tapers=[TW K]; tapers=dpsschk(tapers,Nt,Fs); % get tapers
sig = 5.5; 
% dfs = fcnFindSpSmMTspecRes(f1,f2,f,fp,t,sig,W,findx); 
dfs = fcnFindSpSmMTspecRes2(f1,f2,fp,t,sig,W,findx,tapers);
Ls = fcnFindLeakageSpSmMT2(f1,fp,findx,t,sig,dfs,tapers); 
[Vs,Ss] = fcnFindVarianceSpSmMT(fp,findx,s,t,sig,tapers); 
disp([dfs Ls Vs]);

figure(1); clf; 
plot(fp,pow2db(Sxx),'m--'); 
hold on; 
plot(fp,pow2db(Ss),'b'); 
plot(fp,pow2db(Sm),'r'); 

figure(3); 