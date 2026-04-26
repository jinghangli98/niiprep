% -------------------------------------------------------------------------
%
%                 Rician noise removal via variance stabilization
%
% -------------------------------------------------------------------------
%
% The software implements the algorithm and methods published in the paper:
%  A. Foi, "Noise Estimation and Removal in MR Imaging: the
%  Variance-Stabilization Approach", in Proc. 2011 IEEE Int. Sym.
%  Biomedical Imaging, ISBI 2011, Chicago (IL), USA, April 2011.
%  doi:10.1109/ISBI.2011.5872758
% -------------------------------------------------------------------------

%% INIT
clear all
close all
clc

%% MAIN OPTIONS
% Estimate noise level from data using recursive algorithm with
% VST+Gaussian MAD
estimate_noise = 1;
mixt = [0.25, 0.5, 0.75];
% Denoising algorithm to be used for filtering the variance-stabilized data
denMethod = 'bm4d';      %% BM4D (Maggioni & Foi)

%% load image
addpath('..\NIfTI\');
file_path = '..\Imagens\head_volunteer\';
file_name = 'MPRAGEFA10500V05ISORLGP0InterleavedNormals015a1001';
output_folder = '.\RES\';

nii = load_nii([file_path, file_name, '.nii']);
z = double(nii.img);

fprintf('\n---------------------------------------------------------------\n');
fprintf('Size of data is %d x %d x %d  (total %d voxel)\n', size(z), numel(z));

%% NOISE-LEVEL ESTIMATION
if estimate_noise||~exist('sigma','var')
    fprintf('---------------------------------------------------------------\n');
    fprintf(' * Estimating noise level sigma   [ model  z ~ Rice(nu,sigma) ]\n');
    estimate_noise_printout = 1;   %% print-out estimate at each iteration.
    
    sigma_hat = riceVST_sigmaEst(z, estimate_noise_printout);
    fprintf('---------------------------------------------------------------\n');
    fprintf(' ** sigma_hat = %.4f\n', sigma_hat);
    fprintf('---------------------------------------------------------------\n');
else
    sigma_hat = sigma;
end

%% DENOISING
% VST pair to be used before and after denoising (for forward and inverse
% transformations)
VST_ABC_denoising = 'A';

tic;
fprintf(' * Applying variance-stabilizing transformation\n')
%  Apply variance-stabilizing transformation
fz = riceVST(z, sigma_hat, VST_ABC_denoising);
%  Standard deviation of noise in f(z)
sigmafz = 1;
% BM4D (Maggioni 2011)
fprintf(' * Denoising with Gaussian BM4D ...  (may take a while)\n')
% Apply affine transformation to scale the data to a range well within
% [0,1]. First put data into [0,1], then set data range in [0.15,0.85],
% to avoid clipping of extreme values
maxfz = max(fz(:));
minfz = min(fz(:));
fz = (fz-minfz)/(maxfz-minfz);
sigmafz = sigmafz/(maxfz-minfz); % (scale standard-deviation accordingly)
scale_range = 0.7;
scale_shift = (1-scale_range)/2;
fz = fz*scale_range+scale_shift;
sigmafz = sigmafz * scale_range; % (scale standard-deviation accordingly)
if size(fz,3)>1
    addpath ./bm4d
    % bm4d(z, distribution, sigma, profile, do_wiener, verbose)
    D = bm4d(fz, 'Gauss', sigmafz, 'mp');
else
    addpath ./bm3d
    [dummy, D] = bm3d(1,fz,sigmafz*255,'vn');
end

for i = 1 : length(mixt)
    % VST domain mixture
    DM = D * mixt(i) + fz * (1 - mixt(i));
    % Return filter output to the initial range, applying the inverse
    % affine transformation
    DM = (DM - scale_shift) / scale_range;
    DM = DM * (maxfz - minfz) + minfz;
    
    % Apply exact unbiased inverse for estimating nu
    zm_hat = riceVST_EUI(DM, sigma_hat, VST_ABC_denoising);
    
    % SAVE RESULTS
    nii2 = nii;
    nii2.img = uint16(zm_hat);
    save_nii(nii2, sprintf('%s%s_proc_%.0f.nii', output_folder, file_name, mixt(i)*100));
end

% Return filter output to the initial range, applying the inverse
% affine transformation
D = (D - scale_shift) / scale_range;
D = D * (maxfz - minfz) + minfz;

fprintf(' * Applying exact unbiased inverse for the estimation of nu\n')
% Apply exact unbiased inverse for estimating nu
z_hat = riceVST_EUI(D, sigma_hat, VST_ABC_denoising);

% SAVE RESULTS
nii2 = nii;
nii2.img = uint16(z_hat);
if ~(exist(sprintf('%s%s_proc.nii', output_folder, file_name),'file') == 2)
    save_nii(nii2, sprintf('%s%s_proc.nii', output_folder, file_name));
end

fprintf('   completed in %.2f seconds\n', toc);
fprintf('---------------------------------------------------------------\n');