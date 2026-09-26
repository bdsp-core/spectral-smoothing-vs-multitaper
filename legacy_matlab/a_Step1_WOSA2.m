clear all; clc; format compact; 

%% WOSA [Welch's method]
Fs = 1000;   
t = 0:1/Fs:.296;
x = cos(2*pi*t*200);
% pwelch(x,[],[],[,Fs,'twosided');
NOVERLAP = 1; 
NFFT = nextpow2(length(x));
Fs = 1000;   t = 0:1/Fs:.296;
x = cos(2*pi*t*200)+randn(size(t));
varargin{4}=Fs; 
esttype = 'psd';
[Pxx,F] = welch_mbw(x,esttype,varargin{:});

break
pwelch(x,[],[],[],Fs);

[Pxx,F] = pwelch(x,[],NOVERLAP,NFFT,Fs);
plot(F,Pxx)