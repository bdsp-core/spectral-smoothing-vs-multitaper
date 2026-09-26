function twoPeaks=fcnCheckForTwoPeaksMT(f1,f2,f,S); 

%check whether there are two peaks
[~,ind1]=min(abs(f-f1)); 
y1=pow2db(S(ind1)); 
[~,ind2]=min(abs(f-f2)); 
y2=pow2db(S(ind2)); 
ym=min([pow2db(S(ind1:ind2))]);
twoPeaks= ym<y1 & ym<y2;
