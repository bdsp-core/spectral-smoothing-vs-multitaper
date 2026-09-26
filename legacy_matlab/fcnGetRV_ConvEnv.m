function [x,y,xc,yc] = fcnGetRV_ConvEnv(Lm,Rm,Vm); 

% get (R,V) pairs for fixed L

[~,x]=hist(Lm,25); 


ii{1} = find(Lm>=x(1) & Lm<=x(5)); % low
ii{2} = find(Lm>=x(10) & Lm<=x(15)); % medium
ii{3} = find(Lm>=x(20) & Lm<=x(25)); % high
x=[];

figure(1); clf; 

for i=1:3
    [xt,yt,xct,yct] = fcnGetLowerEnv(ii{i},Rm,Vm); 
    x{i} = xt; y{i} = yt; xc{i} = xct; yc{i} = yct; 
    hold on; 
    plot(xt,yt,'k*',xct,yct); 
end

xlabel('Spectral resolution'); 
ylabel('Variance'); 
set(gcf,'color','w');


%------------------------------------
function [x,y,xc,yc] = fcnGetLowerEnv(ii,Rm,Vm); 

x = Rm(ii); y = Vm(ii); 
xu = unique(x); 
yu = zeros(size(xu)); 
for i=1:length(xu); 
    ind = find(x==xu(i)); 
    yu(i) = min(y(ind)); 
end
x=xu; y=yu; 
ind = find(y>xu(1)); 
x(ind)=[]; y(ind)=[];

if length(x)>2
    k = (convhull(x,y));
else
    k= 1:length(x);
end

xc = x(k(1:end-1)); yc = y(k(1:end-1)); 
