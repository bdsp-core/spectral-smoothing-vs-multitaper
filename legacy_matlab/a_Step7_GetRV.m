clear all; clc; format compact; 

%% MT -- get R,V curve for constant L

%% define signal
T = 10; % Length of timeseries
Fs = 200; % Sampling rate 
signalType = 0; 

% problem -- signal does not have desired characteristics; ignores T
[s,t,f,f1,f2,fp,findx,Sxx] = fcnGetSignal(T,Fs,signalType);

%% get spectral resolution -- MTSA
W=1.5; K=floor(2*T*W-1); 
w = linspace(0.2,5,100); 
k = 1:floor(2*T*W-1); 

ct=0; 
for i=1:length(w); 
    for j=1:length(k); 
        W = w(i); 
        K = k(j); 
        try
            ct=ct+1; 
            [Rm(ct),Lm(ct),Vm(ct)] = fcnGetRLV_MT(W,K,T,Fs,s,t,f1,f2,f,fp,findx);
%             disp([Rm Lm Vm]);
            disp(ct); 
        catch
            disp('fail!'); 
        end
    end
end

%% sort into quintiles

%% sort into high medium low

[x,y,xc,yc] = fcnGetRV_ConvEnv(Lm,Rm,Vm); 

