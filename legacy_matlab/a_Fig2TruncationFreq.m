clear all; clc; format compact; 

%% Illustrate spectral decomposition of EEG signal

%load DATA_Sedation; Fs=200; 
dt=1; Fs=1/dt; Nt=51; 
t=(0:Nt-1)*dt; f=linspace(-Fs/2,Fs/2,Nt); 
s=zeros(size(t)); 
nh=(Nt-1)/2; s(nh-10:nh+10)=1; 

%s=s*0; s(400:500)=30;
Es=sum(s.^2); 
S=fftshift(fft(s)); 
Sm=S; ind=find(abs(f)>.2); Sm(ind)=0; 
E=S.*conj(S); E=E/sum(E)*Es; %E=fftshift(E); 
si =real(ifft(ifftshift(Sm)));

figure(1); clf; 
subplot(211); plot(t,s,'k',t,si,'r'); %axis([0 max(t) -40 40]);  box off
xlabel('Time (sec)','fontsize',12,'fontname','arial');
ylabel('\muv','fontsize',12,'fontname','arial');
set(gca,'fontsize',12,'fontname','arial'); 
set(gca,'tickdir','out'); 

subplot(212); 
xx=[f' flipud(f)']; 
yy=[E' flipud(E)']; 
h=patch(xx,yy,[.5 .5 .5]);
alpha(h,.2); 
set(h,'EdgeColor','none'); hold on; 
plot(f,E,'k',f,E*0,'k','linewidth',1); 
axis([0 20 0 max(E)]); box off
set(gca,'tickdir','out'); 

xlabel('Frequency (Hz)','fontsize',12,'fontname','arial');
ylabel('Energy density log(\muv^2/Hz)','fontsize',12,'fontname','arial');
set(gca,'fontsize',12,'fontname','arial'); 
set(gcf,'color',[1 1 1]); 

%% Get components
figure(2); clf; 
S=fftshift(fft(s)); Sm=S; 
ind=find(f<5 & f>1); ysh=0; bottom=0; prevmin=0; 

subplot(6,1,1); 
plot(t,s,'k'); 
ylabel('\muv','fontsize',12,'fontname','arial');
set(gca,'fontsize',12,'fontname','arial'); 
set(gca,'tickdir','out'); 
set(gca,'xcolor','w'); 

subplot(6,1,[2:5]); 
for i=ind(1:2:end)
    temp=S(i); Si=zeros(size(S)); Si(i)=temp; 
    jdx=find(f>0); ff=f; ff(jdx)=inf;  [~,idx]=min(abs(abs(ff)-f(i)));
    Si(idx)=conj(temp);     
    si =real(ifft(ifftshift(Si))); 
    hold on;
    si=si-max(si); 
    si=si-abs(prevmin)-10;
    plot(t,si,'k'); hold on
    prevmin=min(si); 
    my=mean(si); 
    text(-.8,my,[num2str(f(i)) 'Hz'],'fontsize',9,'fontname','arial');
end
set(gca,'color',[1 1 1],'tickdir','out'); 
set(gcf,'color',[1 1 1]); 
set(gca,'fontsize',12,'fontname','arial');
xlabel('Time (sec)'); 
set(gca,'ycolor','w'); 
%axis off


% Spectrum
subplot(5,1,5); 
xx=[f' flipud(f)']; 
yy=[E' flipud(E)']; 
h=patch(xx,yy,[.5 .5 .5]);
alpha(h,.2); 
set(h,'EdgeColor','none'); hold on; 
plot(f,E,'k',f,E*0,'k','linewidth',1); 
axis([0 20 0 max(E)]); box off
set(gca,'tickdir','out'); 
xlabel('Frequency (Hz)','fontsize',12,'fontname','arial');
ylabel('Energy density ( \muv^2/Hz )','fontsize',12,'fontname','arial');
set(gca,'fontsize',12,'fontname','arial'); 
set(gcf,'color',[1 1 1]); 



set(gcf, 'PaperPositionMode', 'auto');
FileName=['Fig_EEG_SpectralDecomp'];
print('-dpng','-r300',FileName);


