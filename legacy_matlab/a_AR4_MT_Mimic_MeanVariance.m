clear all; clc; format compact;

%% Simulate AR(4) process [46a, Percival and Walden]
Fs=100; Nt=500;  
[s,f,fmt,t,Smt,Sg,Sp]=fcn_Get_MT_Mine(Nt,Fs); 

%% Get PSD estimates
N=100;
SMT=zeros(N,length(Smt)); Sg=zeros(N,length(Sg)); Sp=zeros(N,length(Sp)); 
for i=1:N;
    if ~mod(i,10); disp(i/N); end
    [st,f,fmt,t,Smt,Sg,Sp]=fcn_Get_MT_Mine(Nt,Fs); 
    SMT(i,:)=Smt; SG(i,:)=Sg; SP(i,:)=Sp; 
end
mSMT=mean(SMT); uSMT=quantile(SMT,.75);dSMT=quantile(SMT,.25); vSMT=var(10*log10(SMT)); 
mSG=mean(SG); uSG=quantile(SG,.75); dSG=quantile(SG,.25); vSG=var(10*log10(SG)); 
mSP=mean(SP); uSP=quantile(SP,.75); dSP=quantile(SP,.25); vSP=var(10*log10(SP)); 


%% True PSD
dt=t(2)-t(1); 
den=1-(2.7607*exp(-2*pi*f*dt*j)-3.8106*exp(-2*pi*f*dt*j*2)+2.6535*exp(-2*pi*f*dt*j*3)-0.9238*exp(-2*pi*f*dt*j*4));     
den=den.*conj(den); 
S4=dt./den; 

%% Plot single realization
figure(1); 
ind=find(f>min(fmt)& f<max(fmt));
subplot(211); plot(t,s); 
subplot(212); plot(fmt,10*log10(Smt'),'k',f(ind),10*log10(Sp),f(ind),10*log10(Sg(ind)),'r',f(ind),10*log10(S4(ind)),'m');  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot mean and variance of estimates

%% PLOTS FOR PERIODOGRAM 
figure(2); clf;

subplot(331); plot(f(ind),10*log10(mSP),'r','linewidth',2); hold on; 
plot(f(ind),10*log10(S4(ind)),'k--','linewidth',2);  
[fillhandle,msg]=jbfill(f(ind),10*log10(uSP),10*log10(dSP),[.1 .1 ,.1],[0 0 0],0,.2)
box off; 
set(gca,'fontname','arial','fontsize',15); 
xlabel('Frequency','fontname','arial','fontsize',25)
ylabel('Power','fontname','arial','fontsize',25)

% Bias
bias=abs(10*log10(mSP)-10*log10(S4(ind))); 
subplot(334); 
jbfill(f(ind),bias,0*bias,[.1 .1 ,.1],[0 0 0],0,.2); hold on
plot(f(ind),bias,'k','linewidth',2); 
set(gca,'fontname','arial','fontsize',15); 
xlabel('Frequency','fontname','arial','fontsize',25)
ylabel('10\cdot log_{10}(Bias)','fontname','arial','fontsize',25)
axis([min(fmt) max(fmt) 0 20]); 
box off

% Variance
subplot(337); 
% v=10*log10(vSP); minv=ones(size(v))*min(v); 
v=vSP; minv=ones(size(v))*min(v); 
jbfill(f(ind),v,minv,[.1 .1 ,.1],[0 0 0],0,.2); hold on
plot(f(ind),v,'k','linewidth',2); 
set(gca,'fontname','arial','fontsize',15); 
xlabel('Frequency','fontname','arial','fontsize',25)
ylabel('10\cdot log_{10}(Variance)','fontname','arial','fontsize',25)
axis([min(f(ind)) max(f(ind)) min(minv) 50]); 
box off


% PLOTS FOR MT METHOD
subplot(332); plot(fmt,10*log10(mSMT),'r','linewidth',2); hold on; 
plot(f(ind),10*log10(S4(ind)),'k--','linewidth',2);  
jbfill(fmt,10*log10(uSMT),10*log10(dSMT),[.1 .1 ,.1],[0 0 0],0,.2)
box off
set(gca,'fontname','arial','fontsize',15); 
% xlabel('Frequency','fontname','arial','fontsize',25)
% ylabel('10\cdot log_{10}(Power)','fontname','arial','fontsize',25)

% Bias
S4i=interp1(f(ind),S4(ind),fmt); 
bias=abs(10*log10(mSMT)-10*log10(S4i)); 
for i=10:length(bias); if isnan(bias(i)); bias(i)=bias(i-1); end; end
for i=5:length(bias)-1; if isnan(bias(length(bias)-i));bias(length(bias)-i)=bias(length(bias)-i+1); end; end

subplot(335); 
jbfill(fmt,bias,0*bias,[.1 .1 ,.1],[0 0 0],0,.2); hold on
plot(fmt,bias,'k','linewidth',2); 
box off
set(gca,'fontname','arial','fontsize',15); 
% xlabel('Frequency','fontname','arial','fontsize',25)
% ylabel('10\cdot log_{10}(Bias)','fontname','arial','fontsize',25)
axis([min(fmt) max(fmt) 0 20]); 


% Variance
subplot(338); 
% v=10*log10(vSMT); minv=ones(size(v))*min(v); 
v=vSMT; minv=ones(size(v))*min(v); 
for i=10:length(v); if isnan(v(i)); v(i)=v(i-1); end; end
for i=5:length(v)-1; if isnan(v(length(v)-i));v(length(v)-i)=v(length(v)-i+1); end; end

jbfill(fmt,v,minv,[.1 .1 ,.1],[0 0 0],0,.2); hold on
plot(fmt,v,'k','linewidth',2); 
set(gca,'fontname','arial','fontsize',15); 
xlabel('Frequency','fontname','arial','fontsize',25)
% ylabel('10\cdot log_{10}(Variance)','fontname','arial','fontsize',25)
axis([min(fmt) max(fmt) min(minv) 50]); 
box off

% PLOTS FOR MY METHOD
subplot(333); plot(f(ind),10*log10(mSG(ind)),'r','linewidth',2); hold on; 
plot(f(ind),10*log10(S4(ind)),'k--','linewidth',2);  
[fillhandle,msg]=jbfill(f(ind),10*log10(uSG(ind)),10*log10(dSG(ind)),[.1 .1 ,.1],[0 0 0],0,.2)
box off; 
set(gca,'fontname','arial','fontsize',15); 
% xlabel('Frequency','fontname','arial','fontsize',25)
% ylabel('Power','fontname','arial','fontsize',25)

% Bias
bias=abs(10*log10(mSG)-10*log10(S4)); 
subplot(336); 
jbfill(f(ind),bias(ind),0*bias(ind),[.1 .1 ,.1],[0 0 0],0,.2); hold on
plot(f(ind),bias(ind),'k','linewidth',2); 
box off
set(gca,'fontname','arial','fontsize',15); 
% xlabel('Frequency','fontname','arial','fontsize',25)
% ylabel('10\cdot log_{10}(Bias)','fontname','arial','fontsize',25)
axis([min(fmt) max(fmt) 0 20]); 

% Variance
subplot(339); 
% v=10*log10(vSG(ind)); minv=ones(size(v))*min(v); 
v=vSG(ind); minv=ones(size(v))*min(v); 
jbfill(f(ind),v,minv,[.1 .1 ,.1],[0 0 0],0,.2); hold on
plot(f(ind),v,'k','linewidth',2); 
set(gca,'fontname','arial','fontsize',15); 
xlabel('Frequency','fontname','arial','fontsize',25)
% ylabel('10\cdot log_{10}(Variance)','fontname','arial','fontsize',25)
axis([min(f(ind)) max(f(ind)) min(minv) 50]); 
box off




set(gcf,'color',[1 1 1]); 

