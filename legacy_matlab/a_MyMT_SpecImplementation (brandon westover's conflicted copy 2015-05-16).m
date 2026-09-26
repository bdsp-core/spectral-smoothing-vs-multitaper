clear all; clc; format compact;
load DATA_Spike; Fs=128; s=data(3,:); 
params.Fs=Fs; 
dt=1/Fs; Nt=length(s); t=(0:Nt-1)*dt; 

%% MT estimate
TW=10; K=2*TW-1;
params.tapers=[TW K];
[Smt,fmt]=mtspectrumc(s,params); 

%% Periodogram
f=linspace(-Fs/2,Fs/2,Nt); Es=sum(s.^2); S=fft(s); E=S.*conj(S); E=E/sum(E)*Es; E=fftshift(E); 
ind=find(f>min(fmt)& f<max(fmt)); fi=f(ind); Ei=E(ind); Ei=Ei/sum(Ei)*sum(Smt); 

df=fi(2)-fi(1); ff=df*(-100:100); 
sig=linspace(.1,3,50); 

for i=1:length(sig); 
    g=exp(-(1/2*(ff/sig(i)).^2)); g=g/sum(g); 
    Sg=conv(Ei,g,'same'); 
    figure(1); clf; 
    subplot(211); plot(t,s); 
    subplot(212); plot(fmt,10*log10(Smt),fi,10*log10(Ei),fi,10*log10(Sg),'r');  
    g=input('ok'); 
    %hold on; plot_vector(S,f,[],[],'r')
end


tapers=dpsschk(tapers,N,Fs); % check tapers
J=mtfftc(data,tapers,nfft,Fs);
J=J(findx,:,:);
S=squeeze(mean(conj(J).*J,2));
if trialave; S=squeeze(mean(S,2));end;
if nargout==3; 
   Serr=specerr(S,J,err,trialave);
end;
