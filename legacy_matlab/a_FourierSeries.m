clear all; clc; format compact; 

%load DATA_Sedation; Fs=200; 
load DATA_Spike; Fs=128;
s=data(3,:); dt=1/Fs; Nt=length(s); t=(0:Nt-1)*dt; 
f=linspace(-Fs/2,Fs/2,Nt); 

%s(1:floor(Nt/2))=0; s(floor(Nt/2+1):end)=1; 
s=s*0; s(400:500)=30;
% Generate a signal
% mynoise = noise (Nt, 1, Fs);
% mypeak = peak (Nt, 1, Fs, 5, 115);
% mysignal = -5 * mypeak + 3 * mynoise;
%s=sin(2*pi*1*t); 
%s=mysignal; 
Es=sum(s.^2);
% S=Nt*dt*fft(s); A=S.*conj(S); A=fftshift(A); 
% si =real(1/(Nt*dt)*ifft(S));
S=fft(s); 
Sm=S; ind=find(abs(fftshift(f))>5); Sm(ind)=0; 
E=S.*conj(S); E=E/sum(E)*Es; E=fftshift(E); 
si =real(ifft(Sm));

figure(1); clf; 
subplot(211); plot(t,s,'k',t,si,'r'); axis([0 max(t) -40 40]);  box off
xlabel('Time (sec)','fontsize',12,'fontname','arial');
ylabel('\muv','fontsize',12,'fontname','arial');
set(gca,'fontsize',12,'fontname','arial'); 

subplot(212); semilogy(f,E,'k'); axis([0 max(f) 0 max(E)]); box off
xlabel('Frequency (Hz)','fontsize',12,'fontname','arial');
ylabel('Energy density log(\muv^2/Hz)','fontsize',12,'fontname','arial');
set(gca,'fontsize',12,'fontname','arial'); 
set(gcf,'color',[1 1 1]); 

break
NFFT = 2^nextpow2(Nt); % Next power of 2 from length of y
Y = fft(y,NFFT)/Nt;
f = Fs/2*linspace(0,1,NFFT/2+1);


break
plot(f,2*abs(Y(1:NFFT/2+1)))
title('Single-Sided Amplitude Spectrum of y(t)')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')