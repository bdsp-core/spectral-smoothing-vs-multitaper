clear all; clc; format compact; 

%% 'unpack' the MTSA code
[x,dt,fpass,N,nfft,f,findx,tapers,ff,Fs,Nt,T,W,K,t]=fcnGetStuff;
f1=10; f2=11.5; 

x=sin(2*pi*t*f1)' +  sin(2*pi*t*f2)';  

disp(sprintf('%0.2d, %0.2d, %0.2d, %0.1f, %0.1d',Fs,Nt,T,W,K))

%% MTSA 
for k=1:K
   sw=x.*tapers(:,k); 
   Jk(:,k)=fft(sw,nfft)*dt; 
   Sk(:,k) =conj(Jk(:,k)).*Jk(:,k); % k'th eigenspectrum
end
Sm=mean(Sk,2); 
Sm=Sm(findx); 


%% Smoothing the periodogram
for k=1:K
   temp=fft(tapers(:,k),nfft)*dt; 
   temp=conj(temp).*temp; 
%    temp=fftshift(temp); 
   Hk(:,k)=temp; 
end
H=mean(Hk,2); 

% %-- calculate width of H
% temp = sum(H.^2.*(ff').^2)/sum(H.^2);
% sig_H=4*sqrt(temp) % for a gaussian this has about 95% of "mass" 
% %%

tt=linspace(-.5,.5,length(x))'; sig=0.35; w=exp(-1/2*(tt/sig).^2); w=w/sum(w);
w=papouliswin(length(x));
temp=fft(x.*w,nfft)*dt; 
temp=conj(temp).*temp; 
% temp=fftshift(temp);
Sx=temp;
Ss=real(fft(ifft(H).*ifft(Sx))); 
Ss=Ss(findx); 
Ss=Ss/max(Ss)*max(Sm); 

H=fftshift(H);

%% compare spectra computed by the 2 methods -- something wrong....
figure(2); clf;
plot(f,pow2db(Ss),'r',f,pow2db(Sm))

%% show tapers

figure(1); clf; 
set(gcf,'color','w'); 
% time domain
subplot(311); plot(t,tapers); xlabel('Time [seconds]'); 
text(0,0.69,'DPSS tapers','fontsize',12);

subplot(312); plot(ff,Hk); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,6.5,'Smoothing kernels','fontsize',12);
% show mean of smoothing kernels -- overall smoothing kernel
H=mean(Hk,2); 

subplot(313);
plot(ff,H); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,2.15,'Sum of smoothing kernels','fontsize',12);

figure(2); 

%% Look for valley between two frequencies
figure(2); hold on;
ff1=[f1 f1]; yy=[min([pow2db(Ss); pow2db(Sm)]) max([pow2db(Ss); pow2db(Sm)])];
plot(ff1,yy,'k--');

ff2=[f2 f2]; yy=[min([pow2db(Ss); pow2db(Sm)]) max([pow2db(Ss); pow2db(Sm)])];
plot(ff2,yy,'k--');

%% smoothing windowed periodogram
df=fcnFindSmoothingSpecGramRes(findx,nfft,K,tapers,t,f1,f)

%% mtsa
df=fcnFindMTSAspecRes(findx,nfft,K,tapers,t,f1,f)