% This experiment generate simulated RF data, recompounds them into bmode. 
% The estimated RF data are recomputed by inverting the equations
% and the resulting RF are compared to the original, real ones

%%
clear;
bmode_size = [256 256];
probe = 'P4-2v';
DR = 30;
size_iq = [1986, 64];

% The real max of the absolute IQb signal - this is checked on real data, 
% if unknown set to 1
real_max = 6.2105e+04;

%% Forward path
%% 1a. Simulate the RF data
param = getparam(probe);
param.fs = 4*param.fc; % sampling frequency

% Calculate the transmit delays to generate a non-tilted 80-degrees wide circular wave.
width = 60/180*pi; % width angle in rad
txdel = txdelay(param,0,width); % in s

% Create the scatterers of a 12-cm-by-12-cm phantom.
xs = rand(1,50000)*12e-2-6e-2;
zs = rand(1,50000)*12e-2;
idx = hypot(xs,zs-.05)<1e-2;
xs(idx) = []; % create a 1-cm-radius hole
zs(idx) = [];
RC = 3+randn(size(xs)); % reflection coefficients

% Simulate RF signals by using SIMUS.
RF = simus(xs,zs,RC,txdel,param);


%% 2a. Demodulate the RF signals.

[x,z] = impolgrid(bmode_size,10e-2,pi/3,param);
IQ = rf2iq(RF,param);
% Create a 256x256 80-degrees wide polar grid with IMPOLGRID.

%% 3a. Create the DAS matrix and apply the Delay and sum

Mdas = dasmtx(1i*size(IQ),x,z,txdel,param);
IQb = Mdas*IQ(:);
IQb = reshape(IQb,size(x));
bmode_image = bmode(IQb,DR);

%% Backward path: 
%% 1b. Going from bmode to IQb
IQb_est = bmode2IQb(bmode_image, DR, real_max);

%% 2b. Going from IQb to IQ
IQ_estimated_no_augm = IQb2IQ(IQb_est, Mdas);
IQ_reshaped = reshape(IQ_estimated_no_augm, size_iq);

%% 3b. Experiment 1.1: Plot the estimated and expected IQ data

fig = figure(1);

subplot(1, 2, 1)
plot(real(IQ(:, 50)))
hold on
plot(real(IQ_reshaped(:, 50)))
legend('real','estimated')

subplot(1, 2, 2)
plot(imag(IQ(:, 50)))
hold on
plot(imag(IQ_reshaped(:, 50)))
legend('real','estimated')


%% 4b. Experiment 1.2 Generate and plot expected and simulated bmode image
IQ_reshaped_augm = IQ_reshaped;

IQb_estimated = Mdas * IQ_reshaped_augm(:);
IQb_estimated =  reshape(IQb_estimated,bmode_size);
bmode_estimated = bmode(IQb_estimated,30);

fig = figure(1);

subplot(1, 2, 1)
pcolor(x*100,z*100,bmode_estimated)
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-30 dB','0 dB'};
colormap gray
title('Estimated bmode')
ylabel('[cm]')
shading interp
axis equal ij tight
set(gca,'XColor','none','box','off')

subplot(1, 2, 2)
pcolor(x*100,z*100,bmode_image)
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-30 dB','0 dB'};
colormap gray
title('Real bmode')
ylabel('[cm]')
shading interp
axis equal ij tight
set(gca,'XColor','none','box','off')

%% 5b. Experiment 1.3 Generate and plot expected and simulated bmode image
% when the IQ data are corrupted (some columns are put to 0)

IQ_corrupted = IQ;
IQ_corrupted(:, 1:32) = 0;
IQb = Mdas*IQ_corrupted(:);
IQb = reshape(IQb,size(x));
bmode_image_corrupted = bmode(IQb,DR);

IQ_reshaped_augm = IQ_reshaped;
IQ_reshaped_augm(:, 1:32) = 0;

IQb_estimated = Mdas * IQ_reshaped_augm(:);
IQb_estimated =  reshape(IQb_estimated,bmode_size);
bmode_estimated_corrupted = bmode(IQb_estimated,30);

fig = figure(1);

subplot(1, 2, 1)
pcolor(x*100,z*100,bmode_estimated_corrupted)
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-30 dB','0 dB'};
colormap gray
title('Estimated bmode')
ylabel('[cm]')
shading interp
axis equal ij tight
set(gca,'XColor','none','box','off')

subplot(1, 2, 2)
pcolor(x*100,z*100,bmode_image_corrupted)
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-30 dB','0 dB'};
colormap gray
title('Real bmode')
ylabel('[cm]')
shading interp
axis equal ij tight
set(gca,'XColor','none','box','off')

