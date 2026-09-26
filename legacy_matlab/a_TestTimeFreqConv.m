clear all; clc; format compact; 

N=1001; 
x=randn(1,N); w=randn(1,N);  % random signals

% multiply then take fft
temp1=fftshift(fft(w.*x));
r1=conj(temp1).*temp1; 

% take fft then do convolution
X=(fft(x)); 
W=(fft(w));
temp2=fftshift(fft(ifft(X).*ifft(W))); 
% temp2=conv(X,W,'same'); 

% N=length(X);
% CirculantMatrix = interpMatrix(X,1,N,1,'circ');
% temp2=CirculantMatrix*W(:); %cyclic convolution
% %temp2=ifft(C,'symmetric').'/N
% temp2=temp2'/N;

r2=conj(temp2).*temp2; 

% calculate the discrepency
d=sum(abs(r1-r2))

% plots
subplot(211); 
plot(pow2db(r1)) 

subplot(212); 
plot(pow2db(r2)) 
