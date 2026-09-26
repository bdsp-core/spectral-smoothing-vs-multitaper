function df=fcnFindSmoothingSpecGramRes(findx,nfft,K,tapers,t,f1,f); 

% f1=10;  first freq
Nf=round((max(f)-f1)/(f(2)-f(1)));
f2=linspace(f1,f(end),Nf);  
dt=t(2)-t(1); 
for i=1:length(f2); 
    x=sin(2*pi*t*f1)' +sin(2*pi*t*f2(i))';  
    
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

    % check my method 
    [~,ind1]=min(abs(f-f1)); y1=pow2db(Ss(ind1)); 
    [~,ind2]=min(abs(f-f2(i))); y2=pow2db(Ss(ind2)); 
    ym=min([pow2db(Ss(ind1:ind2))]);
    twoPeaks= ym<y1 & ym<y2;
    if twoPeaks; 
        df=abs(f1-f2(i)); 
        break
    end 
end