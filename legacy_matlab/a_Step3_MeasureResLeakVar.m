clear all; clc; format compact; 

%% fix variance calculation -- do 100 times, measure 1 frequency
%% fix signal normalization

%% define signal
%% create signal with given PSD
T = 1; % Length of timeseries
Fs = 50; % Sampling rate 
Nt = round(Fs*T); % Total number of samples
if ~mod(Nt,2); Nt=Nt+1; end
dt = 1/Fs; t= (0:Nt-1)*dt;                   
Fm=Fs/2; f = linspace(-Fm,Fm,Nt); df= 1/T; fpass=[0 Fm];
[fp,findx]=getfgrid(Fs,Nt,fpass); 
f1=5; 
Nf=round((max(f)-f1)/(f(2)-f(1))); 
f2=linspace(f1,f(end),Nf); dt=t(2)-t(1); 

% get white noise
[s,t,Sxx,f,fp,findx,Nt,T,Fs,dt] = fcnGetSignalAndSxxWhiteNoise;
[s,t,Sxx,f,fp,findx,Nt,T,Fs,dt] = fcnGetSignalAndSxx;

%% get spectral resolution -- MTSA
W=.6; TW=T*W; K=floor(2*TW-1); 
dfm = fcnFindMTSAspecRes(f1,f2,findx,fp,t,f,W,K);
Lm=fcnFindLeakageMT(f1,findx,fp,t,f,W,K,dfm); 
[Vm,Sm]=fcnFindVarianceMT(findx,fp,t,f,W,K,s); 
disp([dfm Lm Vm]);

%% get spectral resolution -- smoothing
W = 0.6; sig = .9; 
dfs = fcnFindSpSmMTspecRes(f1,f2,f,fp,t,sig,W,findx); 
Ls = fcnFindLeakageSpSmMT(f1,t,f,fp,W,K,sig,dfs,findx); 
[Vs,Ss] = fcnFindVarianceSpSmMT(s,t,f,fp,W,K,findx,sig); 
disp([dfs Ls Vs]);

figure(1); clf; plot(log(Sxx),'m--'); hold on; plot(log(Ss),'b'); plot(log(Sm),'r'); 
