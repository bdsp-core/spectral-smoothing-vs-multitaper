clear all; clc; format compact; 
h = 1/3*ones(3,1);
x = randn(64,1);
N = 64+3-1;  %the minimum pad factor
h1 = [h; zeros(N-length(h),1)];
x1 = [x ; zeros(N-length(x),1)];
out = ifft(fft(x1).*fft(h1));
yconv = conv(x,h);