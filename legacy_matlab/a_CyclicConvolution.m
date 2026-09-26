a = 1:5;
b = a+3;
A = fft(a);
B = fft(b);
D = conv(A, B);
d = real(ifft(D))

c = a .* b

%% question: why the "a.*b" is not the same 
% with "real(ifft(D))", is somewhere wrong? 
% I know in the 1st theorem, the Nfft should be 
% length(A)+length(B)-1, however, it seems not work here.

N=length(A);
CirculantMatrix = interpMatrix(A,1,N,1,'circ');
C=CirculantMatrix*B(:); %cyclic convolution
c=ifft(C,'symmetric').'/N
c 