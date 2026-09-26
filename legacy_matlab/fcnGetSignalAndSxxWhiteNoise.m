function [s,t,Sxx,f,fp,findx,Nt,T,Fs,dt] = fcnGetSignalAndSxxWhiteNoise;

%% create signal with given PSD
T = 100; % Length of timeseries
Fs = 50; % Sample rate of timeseries
Nt = round(Fs*T);            % Total number of samples
if ~mod(Nt,2); Nt=Nt+1; end
dt = 1/Fs; t= (0:Nt-1)*dt;                   
Fm=Fs/2; f = linspace(-Fm,Fm,Nt); df= 1/T; 
sig=2;
fpass=[0 Fm];
[fp,findx]=getfgrid(Fs,Nt,fpass); 

% create signal and PSD
Sxx = zeros(size(f))'+1; 
s = fcnGenSignalFromPSD(Sxx,T,df,dt);

%%
T=max(t); 
Sxx=fftshift(Sxx);
Sxx=Sxx(findx);
Sxx = Sxx/sum(Sxx);  