function df=fcnFindMTSAspecRes2(f,f1,f2,fp,findx,t,W,K,tapers); 

% using MTSA
dfMin = inf; 
for i=1:length(f2); 
    s=sin(2*pi*t*f1)' +sin(2*pi*t*f2(i))';  
    
    % spectral res for MTSA
    
    %------------------------------------------------
    Sm = fcnMTSA2(s,t,f,W,K,findx,tapers);
    ind = find(fp>=0); 
    fidx = fp(ind); 
    Sm = Sm(ind); 
    Sm = Sm/sum(Sm);  
    %------------------------------------------------

    %----------------------------
    %check whether two peaks are seperable
    [~,ind1]=min(abs(fidx-f1));    y1=pow2db(Sm(ind1));
    [~,ind2]=min(abs(fidx-f2(i))); y2=pow2db(Sm(ind2));
    
    idx = ind1:ind2;  %find(fidx>=f1 & fidx<=f2(i)); 
    th = linspace(0,1,length(idx)); 
    fy = f1*th+f2(i)*(1-th); 
    y = y2*th+y1*(1-th); 
    c = pow2db(Sm(idx)); 
    

    ih = ceil(length(c)/2); 
    cm = c(ih); 
    ym = y(ih); 

%     figure(1); clf; plot(fidx,pow2db(Sm)); 
%     hold on; 
%     plot(fidx(idx),y,'r',fidx(idx),c,'b','linewidth',2); 
%     plot(fy(ih),ym,'*',fy(ih),cm,'*'); 
%     drawnow
%     g=input('ok'); 
    twoPeaks= cm<ym; 
    %----------------------------
    
    twoPeaks= cm<ym; 

    if twoPeaks==1 & abs(f1-f2(i))<dfMin; 
        dfMin = abs(f1-f2(i)); %disp(dfMin); 
        break
    end
    
end

df = dfMin;