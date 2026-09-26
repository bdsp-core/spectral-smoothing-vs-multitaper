function df = fcnFindSpSmMTspecRes(f1,f2,f,t,sig,W) 

% spectral smoothing using MT kernel -- W as a parameter; does not need to
% match W in MTSA being compared with this
T = max(t); 
TW=T*W; 
K=floor(2*TW-1); 

dfMinMT = inf; 
for i=1:length(f2); 
    s=sin(2*pi*t*f1)' +sin(2*pi*t*f2(i))';  
    
    % get tapers
    [Sm,tapers] = fcnMTSA(s,t,f,W,K);  
    
    %-------------------------
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
    
    ind = find(f>=0); 
    fidx = f(ind); 
    Ss = Ss(ind); 
    
    %-------------------------
    if dfMinMT>10000
        figure(1); clf; plot(fidx,Ss); drawnow
    end
    
    
    %check whether there are two peaks
    [~,ind1]=min(abs(fidx-f1));    y1=pow2db(Ss(ind1))
    [~,ind2]=min(abs(fidx-f2(i))); y2=pow2db(Ss(ind2))
    
    idx = find(f>=f1 & f<=f2(i)); 
    th = linspace(0,1,length(idx)); 
    y = y1*th+y2*(1-th); 
    c = pow2db(Ss(idx)); 
    
    figure(1); clf; plot(fidx,pow2db(Ss),f(idx),y,'r'); drawnow

    keyboard
    
    ym=min([pow2db(Ss(ind1:ind2))])
    twoPeaks= ym<y1 & ym<y2;
    
    
    
    
    
    disp(abs(f1-f2(i))); 
    if twoPeaks==1 & abs(f1-f2(i))<dfMinMT; 
        dfMinMT = abs(f1-f2(i)); disp(dfMinMT); 
    end
end

df = dfMinMT;