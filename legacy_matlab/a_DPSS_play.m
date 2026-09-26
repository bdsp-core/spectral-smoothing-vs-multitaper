clear all; clc; format compact; 
%addpath('C:\Users\Brandon\Documents\My Dropbox\Papers_InProgress\PSG_Matt\Chronux')
addpath('C:\Users\Brandon\Dropbox\Papers_InProgress\PSG_Matt\Chronux'); 
Nt=1281; [E,V]=dpss(Nt,10); t=linspace(0,1,Nt); figure(1); clf; plot(t,E); spE=E;
spE=spE'; 
for i=1:size(spE,1); 
    %load DATA_Spike; Fs=128; s=data(3,:); 
    s=spE(i,:); 
    dt=t(2)-t(1); Fs=1/dt; 
    Nt=length(s); t=(0:Nt-1)*dt; f=linspace(-Fs/2,Fs/2,Nt); 

    
    Es=sum(s.^2); S=fft(s); Sm=S; ind=find(abs(fftshift(f))>5); Sm(ind)=0; 
    E=S.*conj(S); E=E/sum(E)*Es; E=fftshift(E); si =real(ifft(Sm));

    Ss(i,:)=E; 
    
    figure(1); clf; 
    subplot(211); plot(t,s,'k'); axis([0 max(t) min(s) max(s)]);  box off
    xlabel('Time (sec)','fontsize',12,'fontname','arial');
    ylabel('\muv','fontsize',12,'fontname','arial');
    set(gca,'fontsize',12,'fontname','arial'); 

    subplot(212); plot(f,E,'k'); axis([-15 15 0 max(E)]); box off
    xlabel('Frequency (Hz)','fontsize',12,'fontname','arial');
    ylabel('Energy density log(\muv^2/Hz)','fontsize',12,'fontname','arial');
    set(gca,'fontsize',12,'fontname','arial'); 
    set(gcf,'color',[1 1 1]); 
    g=input('ok'); 
end
