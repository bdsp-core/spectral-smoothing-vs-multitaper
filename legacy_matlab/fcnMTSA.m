function [Sm,tapers] = fcnMTSA(s,t,f,W,K,findx);

%% MT estimate
T=max(t); 
Fm=max(f);
Fs=f(2)-f(1); 
x=change_row_to_column(s); % data needs to be column vector
Nt=length(x);
dt=t(2)-t(1);

fpass = [0 Fm]; 
TW=T*W; 
tapers=[TW K];
tapers=dpsschk(tapers,Nt,Fs); % get tapers
ff=linspace(-Fs/2,Fs/2,Nt); 

%% MTSA 
for k=1:K
    sw=x.*tapers(:,k); 
    Jk(:,k)=fft(sw,Nt)*dt; 
    Sk(:,k) =conj(Jk(:,k)).*Jk(:,k); % k'th eigenspectrum
end
Sm=mean(Sk,2); 
Sm=Sm(findx);
Sm=Sm/sum(Sm); 

