%%
clear;

DR = 40;
bmode_real = imread("test_data\test_P4-2V\real_img.png");
img_size = size(bmode_real);
probe = 'P4-2v';

%% 1. Go from bmode to IQb
IQb_est = bmode2IQb(bmode_real, DR);
size_iq = [1986, 64];

%% getting the Mdas matrix based on the probe parameters
param = getparam(probe);
width = 60/180*pi; % width angle in rad
txdel = txdelay(param,0,width); % in s
param.fs = 4*param.fc; % sampling frequency
[x,z] = impolgrid(img_size,10e-2,pi/3,param);

if ~exist('Mdas','var')
    Mdas = dasmtx(1i*size_iq,x,z,txdel,param);
end

%% 2. From the IQb compute the IQ data which corresponds to the envelope
% detected Raw data before the delay and sum
IQ_estimated_no_augm = IQb2IQ(IQb_est, Mdas);
IQ_reshaped = reshape(IQ_estimated_no_augm, size_iq);

%% 4. reconstruct the IQb based on the IQ_estimated corrupted by
% noise/blur/other
IQb_estimated = Mdas * IQ_reshaped(:);
IQb_estimated =  reshape(IQb_estimated,img_size);
bmode_estimated = bmode(IQb_estimated,30);

%% 5. Plot the result
fig = figure(1);

subplot(1, 2, 1)
pcolor(x*100,z*100,bmode_estimated)
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-30 dB','0 dB'};
colormap gray
title('A simulated ultrasound image')
ylabel('[cm]')
shading interp
axis equal ij tight
set(gca,'XColor','none','box','off')

subplot(1, 2, 2)
pcolor(x*100,z*100,bmode_real)
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-30 dB','0 dB'};
colormap gray
title('A simulated ultrasound image')
ylabel('[cm]')
shading interp
axis equal ij tight
set(gca,'XColor','none','box','off')

filename = "test_result/test_result.png";
exportgraphics(figure(1),filename);

