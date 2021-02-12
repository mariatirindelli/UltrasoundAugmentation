% This is a script for using the monogenic signal code for 2D US
% images

% Add monogenic_signal source directory to path
addpath('monogenic_signal_matlab-master/monogenic_signal_matlab-master/src')

x = 1; 
while x <= 2
    % Load the US image
    % Note that the monogenic signal is intended to on greyscale images (using it on
    % a colour image will result in the three channels being processed independently).
    I = imread(sprintf('test_%d.bmp',x));
    [Y,X] = size(I);

    % First we have to choose a set of centre-wavelengths for our filters,
    % typically you will want to play around with this a lot.
    % Centre-wavelengths are expressed in pixel units. Here we use a set of
    % wavelenths with a constant scaling factor of 1.5 between them, starting
    % at 20 pixels
    %create different centre-wavelengths
    pixel_units = [10,15,20,30,40];
    spacing = [0.5, 1.0, 1.5, 3.0];
    i = 1;
    while i <= 5
        j = 1;
        while j <= 4
            cw = pixel_units(i)*spacing(j).^(0:4);
            filtStruct = createMonogenicFilters(Y,X,cw,'lg',0.41);
            [m1,m2,m3] = monogenicSignal(I,filtStruct);
            
            % Local energy (calculated on a per-scale basis)
            LE = localEnergy(m1,m2,m3);
            %figure()
            %imagesc(LE(:,:,1,1)), axis image, axis off, colormap gray
            %title('Local Energy')
            save(sprintf('local_energy_%d_%f_%f.mat',x,pixel_units(i),spacing(j)),'LE')
            j=j+1;
        end
        i=i+1;
    end


    % Local energy (calculated on a per-scale basis)
    %LE = localEnergy(m1,m2,m3);
    %figure()
    %imagesc(LE(:,:,1,1)), axis image, axis off, colormap gray
    %title('Local Energy')
    %save(sprintf('local_energy_%d.mat',x), 'LE')

    % Local phase (calculated on a per-scale basis)
    %LP = localPhase(m1,m2,m3);
    %figure()
    %imagesc(LP(:,:,1,1)), axis image, axis off, colormap gray
    %title('Local Phase')
    %save(sprintf('local_phase_%d.mat',x), 'LP')
    x = x+1
end