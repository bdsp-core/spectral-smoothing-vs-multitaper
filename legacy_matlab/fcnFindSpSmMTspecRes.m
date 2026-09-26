function df = fcnFindSpSmMTspecRes(f1,f2,f,fp,t,sig,W,findx) 

% spectral smoothing using MT kernel -- W as a parameter; does not need to
% match W in MTSA being compared with this
T = max(t); 
TW=T*W; 
K=floor(2*TW-1); 

dfMin = inf; 
for i=1:length(f2); 
    s=sin(2*pi*t*f1)' +sin(2*pi*t*f2(i))';  

    %-------------------------
    [Sm,tapers] = fcnMTSA(s,t,f,W,K,findx);  %/dt/dt; 
    Ss = fcnSmSpect2(s',t,tapers,sig,K,findx);
    ind = find(fp>=0); 
    fidx = fp(ind); 
    Ss = Ss(ind); 
    Ss = Ss/sum(Ss); 
    %-------------------------
        
    %check whether there are two peaks
    [~,ind1]=min(abs(fidx-f1));    y1=pow2db(Ss(ind1));
    [~,ind2]=min(abs(fidx-f2(i))); y2=pow2db(Ss(ind2));
    
    idx = ind1:ind2;  %find(fidx>=f1 & fidx<=f2(i)); 
    th = linspace(0,1,length(idx)); 
    fy = f1*th+f2(i)*(1-th); 
    y = y2*th+y1*(1-th); 
    c = pow2db(Ss(idx)); 
    

%     ym=min([pow2db(Ss(ind1:ind2))]);
    ih = ceil(length(c)/2); 
    cm = c(ih); 
    ym = y(ih); 

%     figure(1); clf; plot(fidx,pow2db(Ss)); 
%     hold on; 
%     plot(fidx(idx),y,'r',fidx(idx),c,'b','linewidth',2); 
%     plot(fy(ih),ym,'*',fy(ih),cm,'*'); 
%     drawnow

    twoPeaks= cm<ym; 

    if twoPeaks==1 & abs(f1-f2(i))<dfMin; 
        dfMin = abs(f1-f2(i)); %disp(dfMin); 
        break
    end
end

df = dfMin;