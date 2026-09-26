clear all; clc; format compact; 

%% 'unpack' the MTSA code
load DATA_Spike; Fs=128; s=data(3,:); 
x=change_row_to_column(s); % data needs to be column vector
x=x(1:end-100); s=s(1:end-100); 
dt=1/Fs; Nt=length(s); t=(0:Nt-1)*dt; 
fpass = [0 64]; 
N=size(x,1); 

%% MT estimate
T=max(t); W=.3; TW=T*W; K=floor(2*TW-1); tapers=[TW K];

disp(sprintf('%0.2d, %0.2d, %0.2d, %0.1f, %0.1d',Fs,Nt,T,W,K))

%% get dpss tapers
nfft=max(2^(nextpow2(N)),N); 
nfft=N;
[f,findx]=getfgrid(Fs,nfft,fpass); 
tapers=dpsschk(tapers,N,Fs); % get tapers
ff=linspace(-Fs/2,Fs/2,nfft); 

%% calculate multitaper spectral estimate -- conventional method (brute force, but transparent; same result as mttfc)
for k=1:K
   sw=x.*tapers(:,k); 
   Jk(:,k)=fft(sw,nfft)*dt; 
   Sk(:,k) =conj(Jk(:,k)).*Jk(:,k); % k'th eigenspectrum
end
Sm=mean(Sk,2); 
Sm=Sm(findx); 

%% show tapers 
figure(1); clf; 
set(gcf,'color','w'); 
% time domain
subplot(311); plot(t,tapers); xlabel('Time [seconds]'); 
text(0,0.69,'DPSS tapers','fontsize',12);

% freq domain -- equivalent smoothing kernel contributed by this taper
for k=1:K
   temp=fft(tapers(:,k),nfft)*dt; 
   temp=conj(temp).*temp; 
   temp=fftshift(temp); 
   Hk(:,k)=temp; 
end

subplot(312); plot(ff,Hk); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,6.5,'Smoothing kernels','fontsize',12);
% show mean of smoothing kernels -- overall smoothing kernel
H=mean(Hk,2); 

subplot(313);
plot(ff,H); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,2.15,'Sum of smoothing kernels','fontsize',12);

%% MTSA -- smoothing kernel apparoach
% show that tapering --> averaging eigenspectra is same as FFT --> smoothing
figure(2); clf

% smoothing method
X=fftshift(fft(x,nfft)*dt); 

Sx=conj(X).*X; 
Ss=conv(Sx,H,'same')*dt;
Ss=ifftshift(Ss); 
Ss=Ss(findx); 

plot(f,pow2db(Sm),f,pow2db(Ss))

% something is wrong with the scaling... 