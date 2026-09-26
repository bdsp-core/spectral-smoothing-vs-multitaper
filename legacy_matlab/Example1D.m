function Example1D
%One dimensional cubic B-Spline fitting example using interpMatrix()
%and comparing different extrapolation schemes.

%%Data

    s = @(t) cos(2*pi*t).*exp(-abs(2*t))+ 2;  %signal to fit

    cubicBspline = @(t) (t>-1 & t<1).*(2/3 - t.^2 +abs(t).^3/2) +...
                        (abs(t)>=1 & abs(t)<2).*((2-abs(t)).^3/6);


    tCoarse=linspace(-1.2, 1.2,9);     %Coarse sample locations on t-axis
       dtCoarse=tCoarse(2)-tCoarse(1);
    tFine=linspace(-1.2, 1.2,81);      %Fine sample locations on t-axis
       dtFine=tFine(2)-tFine(1);

    SampRatio=round(dtCoarse/dtFine); %Sampling ratio

    %sample the signal
    sCoarse=s(tCoarse(:));
    sFine=s(tFine(:));

    figure; subplot(1,2,1) 
    plot(tFine,sFine,'-b', tCoarse,sCoarse,'r*');
    legend('True Signal', 'Control Points');
    title 'Signal Samples'


%%Engine


  kernel=cubicBspline(-2:1/SampRatio:2 );
  nCtrlPts=length(tCoarse);
  
  %create interpolation system matrices
  BasisFine=interpMatrix(kernel, 'max', nCtrlPts, SampRatio, 'mirror');
  BasisCoarse=BasisFine(1:SampRatio:end,:);
  
  %%Do the fit!!! 
  sFit = BasisFine*(BasisCoarse\sCoarse);  

  subplot(1,2,2) 
  hold on;  plot(tFine,sFine,'-b',tFine,sFit,'--m+'); hold off

  %Repeat the above with zero padded extrapolation

  BasisFine=interpMatrix(kernel, 'max', length(tCoarse), SampRatio, 'zero');
  BasisCoarse=BasisFine(1:SampRatio:end,:);
  
  sFit = BasisFine*(BasisCoarse\sCoarse);  
  
  hold on;  plot(tFine,sFit,'-.ko'); hold off


 legend('True Signal', 'Mirror Extrap.','Zero Extrap.')
 title 'Cubic B-Spline Reconstructions'

