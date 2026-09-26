clear all; clc; format compact;

%% Simulate AR(4) process [46a, Percival and Walden]
Fs=100; Nt=800; x=zeros(1,Nt); 
for i=5:Nt; 
    x(i)=2.7607*x(i-1)-3.8106*x(i-2)+2.6535*x(i-3)-0.9238*x(i-4)+randn;
end
s=x(50:end); 
dt=1/Fs; Nt=length(s); t=(0:Nt-1)*dt; 
params.Fs=Fs; 

%% MT estimate
TW=10; K=2*TW-1; params.tapers=[TW K]; tapers=dpsschk([TW K],Nt,Fs); [Smt,fmt]=mtspectrumc(s,params); 

%% Periodogram
f=linspace(-Fs/2,Fs/2,Nt); Es=sum(s.^2); S=fft(s); E=S.*conj(S); E=E/sum(E)*Es; E=fftshift(E); 
ind=find(f>min(fmt)& f<max(fmt)); fi=f(ind); Ei=E(ind); Ei=Ei/sum(Ei)*sum(Smt); 

%% True PSD
w=linspace(-pi,pi,length(f)); 
den=1-(2.7607*exp(-2*pi*f*dt*j)-3.8106*exp(-2*pi*f*dt*j*2)+2.6535*exp(-2*pi*f*dt*j*3)-0.9238*exp(-2*pi*f*dt*j*4));     
den=den.*conj(den); 
S4=dt./den; 

for i=1:19
   temp=fft(tapers(:,i)'); temp=temp.*conj(temp); h(i,:)=fftshift(temp); 
end
Sh=sum(h); Sh=Sh/sum(Sh); 
Sg=conv(Sh,E,'same'); 
Sg=Sg/sum(Sg)*sum(Smt); 


figure(1); 
subplot(211); plot(t,s); 
subplot(212); plot(fmt,10*log10(Smt),'k',fi,10*log10(Ei),'b',f(ind),10*log10(Sg(ind)),'r',f(ind),10*log10(S4(ind)),'m');  

plot(fmt,10*log10(Smt),'k',fi,10*log10(Ei),'b',f(ind),10*log10(S4(ind)),'m')

plot(fi,10*log10(Ei),'b')
break
tapers=dpsschk(tapers,N,Fs); % check tapers
J=mtfftc(data,tapers,nfft,Fs);
J=J(findx,:,:);
S=squeeze(mean(conj(J).*J,2));
if trialave; S=squeeze(mean(S,2));end;
if nargout==3; 
   Serr=specerr(S,J,err,trialave);
end;
