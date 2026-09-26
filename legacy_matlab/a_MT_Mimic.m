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

load TAPERS
for i=1:19
   temp=fft(tapers(:,i)'); temp=temp.*conj(temp); h(i,:)=fftshift(temp); 
end
Sh=sum(h); 
Sg=conv(Sh,E,'same'); 
Sg=Sg/sum(Sg)*sum(Smt); 
subplot(211); plot(t,s); 
subplot(212); plot(fmt,10*log10(Smt),fi,10*log10(Ei),f(ind),10*log10(Sg(ind)),'r');  




break
tapers=dpsschk(tapers,N,Fs); % check tapers
J=mtfftc(data,tapers,nfft,Fs);
J=J(findx,:,:);
S=squeeze(mean(conj(J).*J,2));
if trialave; S=squeeze(mean(S,2));end;
if nargout==3; 
   Serr=specerr(S,J,err,trialave);
end;
