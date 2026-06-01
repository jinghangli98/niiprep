%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Please cite the following papers:
%
%      Campos, V.P. , Santini, T. , Borges, L.R. , Ibrahim, T. S.  and
%      Vieira, MAC "A denoising framework based on variance stabilization 
%      for volumetric magnetic resonance images corrupted with 
%      non-stationary Rician noise", *********
%
%      A. Foi, "Noise Estimation and Removal in MR Imaging: the Variance-Stabilization Approach",
%      in Proc. 2011 IEEE Int. Sym. Biomedical Imaging, ISBI 2011, Chicago (IL), USA, April 2011.
%      doi:10.1109/ISBI.2011.5872758
%
%      M. Maggioni, V. Katkovnik, K. Egiazarian, A. Foi, "A Nonlocal
%      Transform-Domain Filter for Volumetric Data Denoising and
%      Reconstruction", IEEE Trans. Image Process., vol. 22, no. 1, 
%      pp. 119-133, January 2013.  doi:10.1109/TIP.2012.2210725
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Coding implement based on the previous algorithms provided by the
% authors/papers cited above
%
% This work should only be used for nonprofit purposes.
%
% AUTHOR:
%     Vinícius P. Campos, email: vinicius.campos@usp.br
%     Updated - nov/2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [I_denoised]= denoise_with_vst_map(I_noisy, Estimated_NoiseMap, profile, do_wiener, num_threads)

if(~exist('profile', 'var'))
    profile = 'lc';
end

if(~exist('do_wiener', 'var'))
    do_wiener = 1;
end

%******* Deviding noisy image by the Estimated map ****%
I_noisy = I_noisy./Estimated_NoiseMap;

% ************ Aplying Forward Transform ******************** %

I_VST = riceVST(I_noisy,1, 'A');

sigma = 1; %** NOTE -> when data is stabilized, sigma is approx 1 


%****** Rescaling before denoising ******%

max_VST = max(I_VST(:));
min_VST = min(I_VST(:));

I_VST_norm = (I_VST - min_VST)/(max_VST - min_VST); 

sigma =sigma/((max_VST - min_VST));

scale_range = 0.6; scale_shift = 0.2;

I_VST_norm = (I_VST_norm*scale_range + scale_shift); 
sigma = sigma*scale_range;


%****** Denoise with BM4D  ******%


if do_wiener==0
    stage_arg = BM4DProfile.HARD_THRESHOLDING;
else
    stage_arg = BM4DProfile.ALL_STAGES; %choose to perform wiener or not.
end


profile = get_BM4D_profile(profile);

% Limit BM4D mex to the requested number of threads (0 = automatic/all cores)
if(exist('num_threads', 'var') && ~isempty(num_threads))
    profile.num_threads = num_threads;
end


[I_VST_denoised] = BM4D(I_VST_norm,sigma, profile,stage_arg);
    


I_VST_denoised = ((I_VST_denoised) - scale_shift)/scale_range;
I_VST_denoised = I_VST_denoised*(max_VST - min_VST) + min_VST;

% ************ Aplying Inverse Transform ******************** %

I_denoised = riceVST_EUI(I_VST_denoised, 1,'A');

%******* Multiplying denoised image by the Estimated map ****%
I_denoised = I_denoised.*Estimated_NoiseMap;

end
