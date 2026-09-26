clear all; clc; format compact; 

T=10; Nt=501; dt=T/Nt; Fs=1/dt; t=(0:Nt-1)*dt; f=linspace(-Fs/2,Fs/2,Nt); 
t=t-mean(t); 
s=sin(2*pi*t*2)+cos(2*pi*t*3); 
sig=.75; 
w=exp(-1/2*(t/sig).^2); w=w/max(w); 
figure(1); clf; 
subplot(211); plot(t,s,t,w,'r'); 

subplot(212); 
S=fftshift(fft(s.*w)); 
P=S.*conj(S);  
plot(f,P); xlim([0 max(f)])

% check parsevals theorem
powt=sum(s.^2)*dt
powf=sum(P)/Nt*dt

