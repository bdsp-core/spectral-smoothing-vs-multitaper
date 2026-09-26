function L=fcnFindSmoothingSpecGramLeakage(findx,nfft,K,tapers,t,f1,f,df); 

% f1=10;  first freq
Nf=round((max(f)-f1)/(f(2)-f(1)));
f2=linspace(f1,f(end),Nf);  
dt=t(2)-t(1); 

x=sin(2*pi*t*f1)';
    
%% Smoothing the periodogram
for k=1:K
   temp=fft(tapers(:,k),nfft)*dt; 
   temp=conj(temp).*temp; 
%    temp=fftshift(temp); 
   Hk(:,k)=temp; 
end
H=mean(Hk,2); 

tt=linspace(-.5,.5,length(x))'; sig=0.35; w=exp(-1/2*(tt/sig).^2); w=w/sum(w);
w=papouliswin(length(x));
temp=fft(x.*w,nfft)*dt; 
temp=conj(temp).*temp; 
% temp=fftshift(temp);
Sx=temp;
Ss=real(fft(ifft(H).*ifft(Sx))); 
Ss=Ss(findx); 

%% measure leakage: % of power outside of f1+/-df
i0=find(f<=(f1+df) & f>=(f1-df)); 
i1=find(f>(f1+df) | f<(f1-df)); 
Ss=Ss/sum(Ss); 
L=sum(Ss)-sum(Ss(i0)); 
