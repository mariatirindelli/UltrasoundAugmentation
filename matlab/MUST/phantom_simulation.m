
%% Generating the initial image
clear;
density_x = 0;
density_z = 0;

x_spacing = 0.07;
z_spacing = 0.07;

r = 0.5;
x_c = 0;
z_c = 2;

xs_init = [];
zs_init = [];

x_grid = -2:x_spacing:2;
z_grid = 0:z_spacing:4;
%Generating circular scatter 
for i=1:length(x_grid)
for j=1:length(z_grid)
    
    x = x_grid(i);
    z = z_grid(j);
    
    if (x - x_c)^2 + (z - z_c)^2 <= r^2
        xs_init(end+1) = x*1e-2;
        zs_init(end+1) = z*1e-2;
    end
    
end
end

num_frames = 10;
scaling = linspace(1, 3, num_frames);
image_sequence = zeros(200, 200, num_frames);
for i=1:num_frames
%% Get random affine transformation and apply it to the points
T =  [scaling(i) 0 0;
  0 1 0;
  0 0 1];
tform = affine2d(T);
[xs,zs] = transformPointsForward(tform,xs_init,zs_init);

%%
RC = ones(size(xs));
%Select a transducer with GETPARAM
%We want a 128-element linear array.
param = getparam('L11-5v');

%Design the transmit delays with TXDELAY
%The scatterers will be insonified with 21 plane waves steered at -10 to +10 degrees.

tilt = linspace(-10,10,21)/180*pi; % tilt angles in rad
txdel = cell(21,1); % this cell will contain the transmit delays
%Use TXDELAY to calculate the transmit delays for the 21 plane waves.

for k = 1:21
    txdel{k} = txdelay(param,tilt(k));
end
%Check a pressure field with PFIELD
%5Let us visualize the 5th pressure field.

%Define a 100 $\times$ 100 8-cm-by-8-cm grid.

[xi,zi] = meshgrid(linspace(-4e-2,4e-2,100),linspace(0,8e-2,100));
%Simulate the pressure field.

P = pfield(xi,zi,txdel{5},param);
%Display the 5th pressure field.

imagesc(xi(1,:)*1e2,zi(:,1)*1e2,20*log10(P/max(P(:))))
caxis([-30 0]) % dynamic range = [-30,0] dB
c = colorbar;
c.YTickLabel{end} = '0 dB';
colormap([1-hot; hot])
set(gca,'XColor','none','box','off')
axis equal ij
ylabel('[cm]')
title('The 5^{th} plane wave - RMS pressure field')


RF = cell(17,1); % this cell will contain the RF series
param.fs = 4*param.fc; % sampling frequency in Hz

option.WaitBar = false; % remove the progress bar of SIMUS
for k = 1:21
    RF{k} = simus(xs,zs,RC,txdel{k},param,option);
end

%save the RF - this is the raw data 


%This is the 64th RF signal of the 1st series:

rf = RF{1}(:,64);
t = (0:numel(rf)-1)/param.fs*1e6; % time (ms)
plot(t,rf)
set(gca,'YColor','none','box','off')
xlabel('time (\mus)')
title('RF signal of the 64^{th} element (1^{st} series, tilt = -20{\circ})')
axis tight

%Demodulate the RF signals with RF2IQ
%Before beamforming, the RF signals must be I/Q demodulated.

IQ = cell(21,1);  % this cell will contain the I/Q series

for k = 1:21
    IQ{k} = rf2iq(RF{k},param.fs,param.fc);
end
%This is the 64th I/Q signal of the 1st series:

iq = IQ{1}(:,64);
plot(t,real(iq),t,imag(iq))
set(gca,'YColor','none','box','off')
xlabel('time (\mus)')
title('I/Q signal of the 64^{th} element (1^{st} series, tilt = -10{\circ})')
legend({'in-phase','quadrature'})
axis tight


param.fnumber = [];
%Define a 200 $\times$ 200 4-cm-by-4-cm image grid.

[xi,zi] = meshgrid(linspace(-2e-2,2e-2,200),linspace(0,4e-2,200));
%Beamform the I/Q signals using a delay-and-sum with the function DAS.

bIQ = zeros(200,200,21);  % this array will contain the 21 I/Q images

h = waitbar(0,'');
for k = 1:21
    waitbar(k/21,h,['DAS: I/Q series #' int2str(k) ' of 21'])
    bIQ(:,:,k) = das(IQ{k},xi,zi,txdel{k},param);
end
close(h)

%%
save_path = ['RF_compounded_planewaves', int2str(i), '.mat'];
disp(save_path)
save(save_path, 'bIQ');
% save biQ - it's the signal I have to apply echo dechorr to
%%
%Compound ultrasound image
%An ultrasound image is obtained by log-compressing the amplitude of the beamformed I/Q signals. Have a look at the images obtained when steering at -10 degrees.

I = bmode(bIQ(:,:,1),40); % log-compressed image
imagesc(xi(1,:)*1e2,zi(:,1)*1e2,I)
colormap gray
title('PW-based echo image with a tilt angle of -10{\circ}')

axis equal ij
set(gca,'XColor','none','box','off')
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-40 dB','0 dB'};
ylabel('[cm]')

% The individual images are of poor quality. Generate a compound image 
% with the series of 21 diverging waves steered at different angles.

cIQ = sum(bIQ,3); % this is the compound beamformed I/Q

image_sequence(:, :, i) = cIQ;

I = bmode(cIQ,40); % log-compressed image
imagesc(xi(1,:)*1e2,zi(:,1)*1e2,I)
colormap gray
title('Compound PW-based echo image')

axis equal ij
set(gca,'XColor','none','box','off')
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-40 dB','0 dB'};
ylabel('[cm]')
end

save('simulated_rf_compounded.mat', 'image_sequence');
