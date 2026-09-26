function [s,t,f,f1,f2,fp,findx,Sxx] = fcnGetSignal(T,Fs,signalType); 

%% create signal with given PSD
Nt = round(Fs*T); % Total number of samples
if ~mod(Nt,2); Nt=Nt+1; end
dt = 1/Fs; t= (0:Nt-1)*dt;                   
Fm=Fs/2; f = linspace(-Fm,Fm,Nt); df= 1/T; fpass=[0 Fm];
[fp,findx]=getfgrid(Fs,Nt,fpass); 
f1=5; 
Nf=round((max(f)-f1)/(f(2)-f(1))); 
f2=linspace(f1,f(end),Nf); dt=t(2)-t(1); 


if signalType == 0; 
    [s,Sxx] = fcnGetWhiteNoise(T,findx,dt,f,df); 
elseif signalType == 1
    [s,Sxx] = fcnGetColNoiseGauss(T,findx,dt,f,df);
end

% make all into row vectors
Sxx = Sxx';
s = s';

%%-------------------------
% white noise
function [s,Sxx] = fcnGetWhiteNoise(T,findx,dt,f,df); 
% create signal with given PSD
Sxx = zeros(size(f))'+1; 
s = fcnGenSignalFromPSD(Sxx,T,df,dt);
Sxx=fftshift(Sxx);
Sxx=Sxx(findx);
Sxx = Sxx/sum(Sxx); 

%%---------------------------
% colored noise -- gaussian shaped psd
function [s,Sxx] = fcnGetColNoiseGauss(T,findx,dt,f,df);
% create signal and PSD
sig=2;
g=exp(-1/2*(f/sig).^2); g=g/max(g); Sxx=g'; 
Sxx=Sxx.*(g'>.4)+1;
s = fcnGenSignalFromPSD(Sxx,T,df,dt);
Sxx=fftshift(Sxx);
Sxx=Sxx(findx);
Sxx = Sxx/sum(Sxx); 