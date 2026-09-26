function [s,f,fmt,t,Smt,Sg,Sp]=fcn_Get_MT_Mine(Nt,Fs); 

%% Generate realization of AR(4) process & estimates: mine & MT
x=zeros(1,Nt);
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
Sp=Ei;
for i=1:19
   temp=fft(tapers(:,i)'); temp=temp.*conj(temp); h(i,:)=fftshift(temp); 
end
Sh=sum(h); Sh=Sh/sum(Sh); 
Sg=conv(Sh,E,'same'); 
Sg=Sg/sum(Sg)*sum(Smt);