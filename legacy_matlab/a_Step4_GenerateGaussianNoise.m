clear all; clc; format compact; 

%%
T = 100; % Length of timeseries
fs = 25; % Sample rate of timeseries
Nt = round(fs*T);            % Total number of samples
if ~mod(Nt,2); Nt=Nt+1; end
    
dt = 1/fs; t= (0:Nt-1)*dt;                   
Fm=fs/2; f = linspace(-Fm,Fm,Nt); df= 1/T; 

sig=.05; 
g=exp(-1/2*(f/sig).^2); g=g/sum(g); Sxx=g'; 

s = fcnGenSignalFromPSD(Sxx,T,df,dt);

figure(1); clf; 
subplot(211); plot(f,Sxx);
subplot(212); plot(t,s); 