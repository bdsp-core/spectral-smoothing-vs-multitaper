clear all; clc; format compact; 


%% WOSA
N=101; % length of data
s=randn(1,N); s=s/std(s); % signal -- white noise [probably should do in freq domain]
t=linspace(0,1,N); 
dt=t(2)-t(1); 
Fs=1/dt; fm=Fs/2; 
f=linspace(-fm,fm,N); 
Nw=t(10:end); % segment length

% FT matrix
FT=zeros(N,N); 
 for n=1:length(f)
     FT(:,n)=exp(-i*2*pi*f*n)'; 
 end

for i=1:length(Nw)
   K=floor(max(t)/Nw(i)); 
   Sh=zeros(1,N); 
   for j=1:K
      tc=j*Nw(i)-Nw(i)/2; % window center location
      sig=Nw(i)/4; g=exp(-1/2*((t-tc)/sig).^2); w=g/sum(g); % gaussian window
      figure(1); clf; subplot(211); 
      plot(t,g); 
      
      ws=s.*w; 
      Sh=Sh+(ws*FT).*conj(ws*FT); 
   end
   Sh=Sh/K; 
   subplot(212); plot(f,Sh); drawnow; %g=input('ok')
    
end