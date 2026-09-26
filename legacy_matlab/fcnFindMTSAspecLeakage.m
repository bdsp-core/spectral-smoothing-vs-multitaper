function L=fcnFindMTSAspecLeakage(findx,nfft,K,tapers,t,f1,f,df); 

% f1=10;  first freq
Nf=round((max(f)-f1)/(f(2)-f(1)));
f2=linspace(f1,f(end),Nf);  
dt=t(2)-t(1); 
 
x=sin(2*pi*t*f1)' ;  
for k=1:K
   sw=x.*tapers(:,k); 
   Jk(:,k)=fft(sw,nfft)*dt; 
   Sk(:,k) =conj(Jk(:,k)).*Jk(:,k); % k'th eigenspectrum
end
Sm=mean(Sk,2); 
Sm=Sm(findx);
    
%% measure leakage: % of power outside of f1+/-df
i0=find(f<=(f1+df) & f>=(f1-df)); 
i1=find(f>(f1+df) | f<(f1-df)); 
Sm=Sm/sum(Sm); 
L=sum(Sm)-sum(Sm(i0)); 