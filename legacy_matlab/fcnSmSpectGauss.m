function [Ss,H,Hk] = fcnSmSpectGauss(s,t,tapers,sig,K);

Nt=length(t); 
for k=1:K
   temp=fft(tapers(:,k),Nt); 
   Hk(:,k)=conj(temp).*temp;  
end
H=mean(Hk,2); 
tt=linspace(-.5,.5,length(s))'; 

% windowed fft
w=exp(-1/2*(tt/sig).^2); w=w/sum(w);
temp=fft(s.*w,Nt); 

% convolve with H
Sx=conj(temp).*temp;
Ss=real(fft(ifft(H).*ifft(Sx))); 
