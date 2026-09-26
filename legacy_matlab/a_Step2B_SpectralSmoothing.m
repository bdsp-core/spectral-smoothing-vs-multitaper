clear all; clc; format compact; 

%% check spectral resolution 

%% define signal
%% create signal with given PSD
T = 10; % Length of timeseries
Fs = 50; % Sampling rate 
Nt = round(Fs*T); % Total number of samples
if ~mod(Nt,2); Nt=Nt+1; end
dt = 1/Fs; t= (0:Nt-1)*dt;                   
Fm=Fs/2; f = linspace(-Fm,Fm,Nt); df= 1/T; fpass=[0 Fm];
[fp,findx]=getfgrid(Fs,Nt,fpass); 
f1=10; Nf=round((max(f)-f1)/(f(2)-f(1))); 
f2=linspace(f1,f(end),Nf); dt=t(2)-t(1); 

%% get spectral resolution -- smoothing
W = 0.5; sig = 0.5; 

T = max(t); 
TW=T*W; 
K=floor(2*TW-1); 

dfMinMT = inf; 

s=sin(2*pi*t*f1)' +sin(2*pi*t*f2(1))';  
    
% get tapers
[Sm,tapers] = fcnMTSA(s,t,f,W,K);  

%-------------------------
Nt=length(t); 
for k=1:K
   temp=fft(tapers(:,k),Nt); 
   Hk(:,k)=conj(temp).*temp;  
end
H=mean(Hk,2); 
tt=linspace(-.5,.5,length(s))'; 

% windowed fft
w=exp(-1/2*(tt/sig).^2); w=w/sum(w);
temp=fft(s.*w,Nt); 

% convolve with H
Sx=conj(temp).*temp;
Ss=ifftshift(real(fft(ifft(H).*ifft(Sx)))); 

ind = find(f>=0); 
fidx = f(ind); 
Ss = Ss(ind); 

%check whether there are two peaks
[~,ind1]=min(abs(fidx-f1));    y1=pow2db(Ss(ind1))
[~,ind2]=min(abs(fidx-f2(1))); y2=pow2db(Ss(ind2))

figure(1); clf; plot(fidx,pow2db(Ss)'); 
