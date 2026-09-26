function [s,f,t,Sp]=fcn_Get_SP(Nt,Fs); 

%% Generate realization of AR(4) process & estimates: mine & MT
x=zeros(1,Nt); x(1:3)=randn(1,3);  dt=1/Fs; t=(0:Nt-1)*dt;
for i=5:Nt; 
    x(i)=2.7607*x(i-1)-3.8106*x(i-2)+2.6535*x(i-3)-0.9238*x(i-4)+randn;
    x(i)=x(i)+400*sin(t(i)*2*pi*(25+.1*randn)+.01*randn);
    x(i)=x(i)+500*sin(t(i)*2*pi*(20+.1*randn)+.01*randn);
end
s=x; 
Nt=length(s); ; 

%% Periodogram
f=linspace(-Fs/2,Fs/2,Nt); Es=sum(s.^2); S=fft(s); E=S.*conj(S); E=E/sum(E)*Es; E=fftshift(E); 
ind=find(f>0 & f<max(f)); Ei=E(ind); 
Sp=Ei;