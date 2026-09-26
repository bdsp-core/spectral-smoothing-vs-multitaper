function Example2D
%Two dimensional cubic B-Spline fitting example using interpMatrix() 
%in conjunction with KronProd().


%%Data

    s = @(t) cos(2*pi*t).*exp(-abs(2*t))+ 1;  %signal to fit

    cubicBspline = @(t) (t>-1 & t<1).*(2/3 - t.^2 +abs(t).^3/2) +...
                        (abs(t)>=1 & abs(t)<2).*((2-abs(t)).^3/6);


    tCoarse=linspace(-1.2, 1.2,9);     %Coarse sample locations on t-axis
       dtCoarse=tCoarse(2)-tCoarse(1);
    tFine=linspace(-1.2, 1.2,81);      %Fine sample locations on t-axis
       dtFine=tFine(2)-tFine(1);

    SampRatio=round(dtCoarse/dtFine); %Sampling ratio

    %sample the signal
    sCoarse1D=s(tCoarse(:));  sCoarse1D=sCoarse1D;
    sFine1D=s(tFine(:));      sFine1D=sFine1D;

    sCoarse=sCoarse1D*sCoarse1D.';
    sFine=sFine1D*sFine1D.';
  
    figure; subplot(121)
    surf(tFine,tFine,sFine);

    title 'Signal Samples'
    
%%Engine

  kernel=cubicBspline(-2:1/SampRatio:2 );
  nCtrlPts=length(tCoarse);
  
  %create interpolation system matrices

  Basis1DFine=interpMatrix(kernel, 'max', nCtrlPts, SampRatio, 'mirror');
  BasisFine=KronProd({Basis1DFine}, [1 1]);
  
  
  Basis1DCoarse=Basis1DFine(1:SampRatio:end,:);
  BasisCoarse=KronProd({Basis1DCoarse}, [1 1]);
  
     %Alternatively, we could have used KronProd class cellfun() method:
     %
     %   BasisCoarse=cellfun(@(A) A(1:SampRatio:end,:) ,BasisFine);
  
     
     
  %%Do the fit!!! 
  sFit = BasisFine*(BasisCoarse\sCoarse);  

  subplot(122)
  surf(tFine,tFine,sFit);
  title 'Cubic B-Spline Reconstructions'
  
  

