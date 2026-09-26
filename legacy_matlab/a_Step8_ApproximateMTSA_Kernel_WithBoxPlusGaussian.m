clear all; clc; format compact; 

%% 'unpack' the MTSA code
[x,dt,fpass,N,nfft,f,findx,tapers,ff,Fs,Nt,T,W,K,t]=fcnGetStuff;
disp(sprintf('%0.2d, %0.2d, %0.2d, %0.1f, %0.1d',Fs,Nt,T,W,K))

%% Smoothing the periodogram
for k=1:K
   temp=fft(tapers(:,k),nfft)*dt; 
   temp=conj(temp).*temp; 
   temp=fftshift(temp); 
   Hk(:,k)=temp; 
end
H=mean(Hk,2); 
H = (H/sum(H))'; 

%-- calculate width of H
% temp = sum(H.^2.*(ff').^2)/sum(H.^2);
% sig_H=4*sqrt(temp) % for a gaussian this has about 95% of "mass" 
%% try to approximate the mtsa kernel with papoulis taper convolved with square wave

% get fft of papoulis window (squared)
tt=linspace(-.5,.5,length(x))'; sig=0.4; w=exp(-1/2*(tt/sig).^2); w=w/sum(w);
w=papouliswin(length(x));
temp = fft(w,nfft)*dt; 
temp = (conj(temp).*temp)'; 
Wp = temp;
% Wp   = fftshift(temp)'; 

% convolve with square wave in frequency domain
Ws = zeros(size(ff)); 

width = linspace(0.1,1,1000); 
Emin=inf; 
for i = 1:length(width)
    ind = find(abs(ff)<=width(i)); 
    Ws = zeros(size(ff)); 
    Ws(ind) = 1; 
    Ws = ifftshift(Ws); 
    temp=real(fft(ifft(Ws).*ifft(Wp))); 
    Wa   = fftshift(temp); 
    Wa = Wa/sum(Wa); 
    
    E(i) = sum((Wa-H).^2); 
    disp([E(i),i]); 
    
    if E(i)<Emin; 
        figure(2); clf; plot(ff,H,ff,Wa); xlim([-5 5]); drawnow; 
        Emin=E(i); 
    end
%     g = input('ok'); 
end

% Wa=Wa/max(Wa)*max(Wa); 

%% show tapers
return
figure(1); clf; 
set(gcf,'color','w'); 
% time domain
subplot(311); plot(t,tapers); xlabel('Time [seconds]'); 
text(0,0.69,'DPSS tapers','fontsize',12);

subplot(312); plot(ff,Hk); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,6.5,'Smoothing kernels','fontsize',12);
% show mean of smoothing kernels -- overall smoothing kernel
% H=mean(Hk,2); 

% plot MTSA kernel
subplot(313);
plot(ff,H); xlim([-5 5]); xlabel('Frequency [Hz]'); 
text(-5,2.15,'Sum of smoothing kernels','fontsize',12);

