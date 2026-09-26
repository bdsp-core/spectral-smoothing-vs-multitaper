function Ss = fcnSmSpect2(s,t,tapers,sig,findx);

K = size(tapers,2); 
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
Sx=conj(temp).*temp; ;
Ss=real(fft(ifft(H).*ifft(Sx))); 

Ss=Ss(findx); 
Ss = Ss';
Ss=Ss/sum(Ss); 

figure(3); clf; 
subplot(131); plot(tt,s,tt,w/max(w)*max(s)); 
subplot(132); plot(tt,s.*w);  

fx = fft(s,Nt); fx = conj(fx).*fx; fx = fx(findx); fx = fx/sum(fx); 
sx = Sx(findx); sx = sx/sum(sx); 
subplot(133); plot(pow2db(sx)); hold on; plot(pow2db(fx)); 

