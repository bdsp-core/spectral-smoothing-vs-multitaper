function [x,dt,fpass,N,nfft,f,findx,tapers,ff,Fs,Nt,T,W,K,t]=fcnGetStuff

load DATA_Spike; Fs=128; s=data(3,:); 
x=change_row_to_column(s); % data needs to be column vector
x=x(1:end-100); s=s(1:end-100); 
dt=1/Fs; Nt=length(s); t=(0:Nt-1)*dt; 

% ti=linspace(0,max(t),5*Nt); 
% si=interp1(t,s,ti); 
% t=ti; s=si; x=s'; 

dt=t(2)-t(1); Fs=1/dt; 
Nt=length(s); t=(0:Nt-1)*dt; 
fpass = [0 64]; 
N=size(x,1); 

%% MT estimate
T=max(t); 
T=2; 
W=.5; TW=T*W; K=floor(2*TW-1); tapers=[TW K];

nfft=max(2^(nextpow2(N)),N); 
nfft=N;
[f,findx]=getfgrid(Fs,nfft,fpass); 
tapers=dpsschk(tapers,N,Fs); % get tapers
ff=linspace(-Fs/2,Fs/2,nfft); 
