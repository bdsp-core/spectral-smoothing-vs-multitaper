clear all; clc; format compact; 

%% check spectral resolution 

%% define signal
%% create signal with given PSD
T = 20; % Length of timeseries
Fs = 50; % Sampling rate 
Nt = round(Fs*T); % Total number of samples
if ~mod(Nt,2); Nt=Nt+1; end
dt = 1/Fs; t= (0:Nt-1)*dt;                   
Fm=Fs/2; f = linspace(-Fm,Fm,Nt); df= 1/T; fpass=[0 Fm];
[fp,findx]=getfgrid(Fs,Nt,fpass); 
f1=10; Nf=round((max(f)-f1)/(f(2)-f(1))); 
f2= f1:dt:f(end); 


%% get spectral resolution -- MTSA
W=.3; TW=T*W; K=floor(2*TW-1); 
df = fcnFindMTSAspecRes(f1,f2,findx,fp,t,f,W,K);
L=fcnFindLeakageMT(f1,findx,fp,t,f,W,K,df); 


%% get spectral resolution -- smoothing
% W = 0.3; sig = 0.5; 
% df = fcnFindSpSmMTspecRes(f1,f2,f,t,sig,W)


