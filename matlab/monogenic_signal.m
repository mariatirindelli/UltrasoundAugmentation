% This is a script for using the monogenic signal code for 2D US
% images

% Add monogenic_signal source directory to path
addpath('monogenic_signal_matlab-master/monogenic_signal_matlab-master/src')

x = 1; 
while x <= 12
    % Load the US image
    % Note that the monogenic signal is intended to on greyscale images (using it on
    % a colour image will result in the three channels being processed independently).
    %I = imread(sprintf('test_%d.bmp',x));
    I = imread(sprintf('test_US_%d.bmp',x));
    [Y,X] = size(I);

% First we have to choose a set of centre-wavelengths for our filters,
% typically you will want to play around with this a lot.
% Centre-wavelengths are expressed in pixel units. Here we use a set of
% wavelenths with a constant scaling factor of 1.5 between them, starting
% at 20 pixels
%cw = 20*1.5.^(0:4);
cw = 15*1.0.^(0:4);

% Now use these wavelengths to create a structure containing
% frequency-domain filters to calculate the mnonogenic signal. We can
% re-use this structure many times if we need for many images of the same
% size and using the same wavelength. We can choose from a number of
% different filter types, with log-Gabor ('lg') being the default. For lg
% filters we can also choose the shape parameter (between 0 and 1), which
% governs the bandwidth (0.41 gives a three-octave filter, 0.55 gives a two
% octave filter)
% filtStruct = createMonogenicFilters(Y,X,cw,'lg',0.55);
filtStruct = createMonogenicFilters(Y,X,cw,'lg',0.41);


% Now we can use this structure to find the monogenic signal for the image
[m1,m2,m3] = monogenicSignal(I,filtStruct);

% The returned values are the three parts of the monogenic signal: m1 is
% the even part, and m2 and m3 are the odd parts in the vertical and
% horizontal directions respectively. Each array is Y x X x 1 x W, where
% X and Y are the image dimensions and W is the number of wavelengths.
% The filter responses to the filters of each scale are stacked along the
% fourth dimension.

% From here we can straightforwardly find many of the derived measures by
% passing these three arrays

% Local energy (calculated on a per-scale basis)
LE = localEnergy(m1,m2,m3);
%figure()
%imagesc(LE(:,:,1,1)), axis image, axis off, colormap gray
%title('Local Energy')
save(sprintf('test_result/local_energy_US_%d.mat',x), 'LE')
%impixelinfo
% Local phase (calculated on a per-scale basis)
LP = localPhase(m1,m2,m3);
%figure()
%imagesc(LP(:,:,1,1)), axis image, axis off, colormap gray
%title('Local Phase')
save(sprintf('test_result/local_phase_US_%d.mat',x), 'LP')
%impixelinfo
% Feature symmetry and asymmetry (see Kovesi "Symmetry and Asymmetry from
% Local Phase") pick out 'blob-like' and 'boundary-like' structures
% respectively. This combines all the scales to give a single 2D image.
[FS,FA] = featureSymmetry(m1,m2,m3);
%figure()
%imagesc(FS), axis image, axis off, colormap gray
%title('Feature Symmetry')
save(sprintf('test_result/feature_symmetry_US_%d.mat',x), 'FS')
%impixelinfo
x = x+1;
end

