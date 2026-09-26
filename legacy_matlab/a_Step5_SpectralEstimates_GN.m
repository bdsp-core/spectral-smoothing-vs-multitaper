clear all; clc; format compact; 

%% to do: 
% make f consistent for entire code -- don't overwrite 
% plots of spect res vs leakage, vice versa
% plots of variance vs spect leakage, res

[s,t,Sxx,f,fp,findx,Nt,T,Fs,dt] = fcnGetSignalAndSxx;
% [s,t,Sxx,f,fp,findx,Nt,T,Fs,dt] = fcnGetSignalAndSxxWhiteNoise;

W=.51; TW=T*W; K=floor(2*TW-1); 
[Sm,tapers] = fcnMTSA(s,t,f,W,K,findx);  %/dt/dt; 


%% Smoothing the periodogram
sig=0.8; 
[Ss,H,Hk] = fcnSmSpect(s,t,tapers,sig,K,findx);  %/dt/dt; 


%%

H=fftshift(H); 

%% compare spectra computed by the 2 methods -- something wrong....
figure(2); clf;
plot(fp,pow2db(Ss),'r',fp,pow2db(Sm),fp,pow2db(Sxx),'k--')

%% show tapers

figure(1); clf; 
set(gcf,'color','w'); 
% time domain
subplot(311); plot(t,tapers); xlabel('Time [seconds]'); 
text(0,0.69,'DPSS tapers','fontsize',12);

subplot(312); plot(f,Hk); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,6.5,'Smoothing kernels','fontsize',12);
% show mean of smoothing kernels -- overall smoothing kernel

subplot(313);
plot(f,H); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,2.15,'Sum of smoothing kernels','fontsize',12);

figure(2); 