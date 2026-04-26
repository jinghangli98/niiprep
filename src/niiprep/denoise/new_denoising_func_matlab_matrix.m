%Denoising software
%adapted from Vinicius (out) and Fabricio's (out_old) code (From Dr. Marcelo Vieiras Lab, University of Sao Paulo, Brazil)
%Tales Santini, Aug 8 2018
%usage [out, out_old] =denoising_func_matlab_matrix(matrix)

function [out, out_old] = new_denoising_func_matlab_matrix(matrix, type_var)

p = mfilename('fullpath');
[denoising_script_folder,~,~] = fileparts(p);
%settings:
if isempty(type_var)
    type_var = class(matrix);
end

addpath(genpath([denoising_script_folder filesep 'RiceOptVST']));

%original_size = size(matrix);
%if length(original_size) == 4
%    matrix = reshape(matrix,original_size(1),original_size(2),[]);
%end


z = double(matrix);
denoising_weight = 1;
txt_str = '';

fprintf('%s * Estimating noise level sigma   [ model  z ~ Rice(nu,sigma) ]\n', txt_str);

sigma_hat = riceVST_sigmaEst(z, 1);

fprintf('%s   |-->   sigma_hat = %.4f', txt_str, sigma_hat);

fprintf('%s * Applying variance-stabilizing transformation\n', txt_str);

VST_ABC_denoising = 'A';
fz = riceVST(z, sigma_hat, VST_ABC_denoising);
sigmafz = 1;

fprintf('%s * Denoising with Gaussian BM4D\n', txt_str);


% Apply affine transformation to scale the data to a range well within [0,1]
% First put data into [0,1]
maxfz = max(fz(:));
minfz = min(fz(:));
fz = (fz - minfz) / (maxfz - minfz);
% (scale standard-deviation accordingly)
sigmafz = sigmafz / (maxfz - minfz);
% Then set data range in [0.15,0.85], to avoid clipping of extreme values
scale_range = 1;
scale_shift = (1 - scale_range) / 2;
fz = fz * scale_range + scale_shift;
% (scale standard-deviation accordingly)
sigmafz = sigmafz * scale_range;

%D = bm4d(fz, 'Gauss', sigmafz, 'mp');
D = bm4d(fz, 'Gauss', 0, 'mp');

% Return filter output to the initial range, applying the inverse affine transformation
D = (D - scale_shift) / scale_range;
D = D * (maxfz - minfz) + minfz;

fprintf('%s * Applying exact unbiased inverse for the estimation of nu\n', txt_str);

% Apply exact unbiased inverse for estimating nu
nu_hat = riceVST_EUI(D, sigma_hat, VST_ABC_denoising);

S.nu_hat = nu_hat;

w_val = denoising_weight;
out_old = eval([type_var '(nu_hat * w_val + z * (1 - w_val));']);

%if length(original_size) == 4
%    out = reshape(out, original_size);
%end

fprintf('%s * w = %.2f / SNR Improvement: %.2f x\n', txt_str, w_val,  1/(1 - w_val));
%% Vinicius implementation

addpath(genpath([denoising_script_folder filesep 'Dmri_LAVI']));

I_noisy_aux = double(matrix);
lpf_sigma=4;

for i =1:size(I_noisy_aux,4)

   [Sigma_map_rician(:,:,:,i), ~] = rice_sigma_mapEST(I_noisy_aux(:,:,:,i),0,lpf_sigma);

   [I_denoised_our_prop(:,:,:,i)]=denoise_with_vst_map(I_noisy_aux(:,:,:,i),Sigma_map_rician(:,:,:,i),'mp',0,1,1,0);
%toc
end   

out = eval([type_var '(I_denoised_our_prop*32767/max(I_denoised_our_prop(:)));']);

end
