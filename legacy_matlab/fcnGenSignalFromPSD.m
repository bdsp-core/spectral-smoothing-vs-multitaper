function s = fcnGenSignalFromPSD(P,T,df,dt)

P=ifftshift(P);
N=length(P);
Xm_mag = sqrt(P.*T); 

nh= (N-1)/2; 
randnums = rand(nh,1).*2*pi;  % Random phase between 0 and 2pi
randvalues = exp(i*randnums);   % white noise
WN = [1; randvalues; flip(conj(randvalues))];
S= Xm_mag.*WN; % [meters]
s = real(ifft(S)*N*df)/dt;