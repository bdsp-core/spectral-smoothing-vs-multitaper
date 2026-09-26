clear all; clc; format compact;

%% Vary length of signal, calculate mean, variance

%% Simulate AR(4) process [46a, Percival and Walden]
Fs=100; Nt=500;  
Ntt=[50 100 200];
figure(2); clf;

for kk=1:length(Ntt)
    Nt=Ntt(kk); 
    [s,f,t,Sp]=fcn_Get_SP(Nt,Fs); 

    %% Get PSD estimates
    N=100;
    SP=zeros(N,length(Sp)); 
    for i=1:N;
        if ~mod(i,10); disp(i/N); end
        [s,f,t,Sp]=fcn_Get_SP(Nt,Fs); 
        SP(i,:)=Sp; 
    end
    mSP=mean(SP); uSP=quantile(SP,.75); dSP=quantile(SP,.25); vSP=var(10*log10(SP)); 

    %% True PSD
    dt=t(2)-t(1); 
    den=1-(2.7607*exp(-2*pi*f*dt*j)-3.8106*exp(-2*pi*f*dt*j*2)+2.6535*exp(-2*pi*f*dt*j*3)-0.9238*exp(-2*pi*f*dt*j*4));     
    den=den.*conj(den); 
    S4=dt./den; 

    %% Plot single realization
%     figure(1); 
     ind=find(f>0& f<max(f));
%     subplot(211); plot(t,s); 
%     subplot(212); plot(fmt,10*log10(Smt'),'k',f(ind),10*log10(Sp),f(ind),10*log10(Sg(ind)),'r',f(ind),10*log10(S4(ind)),'m');  

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plot mean and variance of estimates

    %% PLOTS FOR PERIODOGRAM 

    subplot(3,1,kk); plot(f(ind),10*log10(mSP),'r','linewidth',2); hold on; 
    plot(f(ind),10*log10(S4(ind)),'k--','linewidth',2);  
    [fillhandle,msg]=jbfill(f(ind),10*log10(uSP),10*log10(dSP),[.1 .1 ,.1],[0 0 0],0,.2)
    box off; 
    set(gca,'fontname','arial','fontsize',15); 
    xlabel('Frequency','fontname','arial','fontsize',25)
    ylabel('Power','fontname','arial','fontsize',25)
    set(gcf,'color',[1 1 1]); 
    
end

