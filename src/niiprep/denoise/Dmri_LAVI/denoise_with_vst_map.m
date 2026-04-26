%************************* DENOISING ONLY AFTER VST ********************%

function [I_denoised, sigma_map_BM4D]= denoise_with_vst_map(I_noisy, sigma_map, profile, bm4d_alone,do_wiener , do_previous, I_ref)

if(~exist('profile', 'var'))
    profile = 'np';
end

if(~exist('bm4d_alone', 'var'))
    bm4d_alone = 0;
end

if(~exist('do_wiener', 'var'))
    do_wiener = 1;
end


%************** ESTIMATED MAPS(FERNANDEZ) + FOI'S VST + BM3D *******%

%************ Sigma map estimation ************%


%**** Dividing I_noisy by the estimated map (removing spatial-dependece)


% = I_noisy./sqrt(L/2);
%est_map_quadrado = (Estimated_NoiseMap.^2)./2;


if bm4d_alone==0

sigma_map_BM4D = zeros(size(I_noisy));

Estimated_NoiseMap = sigma_map;
I_noisy = I_noisy./Estimated_NoiseMap;
% ************ Aplying Forward Transform ******************** %

I_VST = riceVST(I_noisy,1, 'B');
%function_stdEst(I_noisy,4,1,-1);
sigma = 1; %** NOTE -> when data is stabilized, sigma is approx 1 


%****** Rescaling before denoising ******%

max_VST_FOI = max(I_VST(:));
min_VST_FOI = min(I_VST(:));

I_VST_norm = (I_VST - min_VST_FOI)/(max_VST_FOI - min_VST_FOI); 

sigma =sigma/((max_VST_FOI - min_VST_FOI));

scale_range = 0.4; scale_shift = 0.1;

I_VST_norm = (I_VST_norm*scale_range + scale_shift); 
sigma = sigma*scale_range;


max_I_ref = max(I_ref(:));
min_I_ref = min(I_ref(:));

I_ref_norm = (I_ref - min_I_ref)/(max_I_ref - min_I_ref); 
I_ref_norm = (I_ref_norm*scale_range + scale_shift); 


%****** Denoise with BM3D/BM4d ******%
if size(I_VST_norm,3) > 1
     
     %bm4d(z, distribution, sigma, profile, do_wiener, verbose)
     if(size(I_VST,4))>1
         sigma = sigma.*ones(1,size(I_VST_norm,4));
     end
     [I_VST_denoised, ~] = bm4d(I_VST_norm,'Gauss', sigma, profile,1,0);
    
else
    [~, I_VST_denoised] = BM3D(1, I_VST_norm, sigma*255, profile,0);
end

I_VST_denoised = ((I_VST_denoised) - scale_shift)/scale_range;
I_VST_denoised = I_VST_denoised*(max_VST_FOI - min_VST_FOI) + min_VST_FOI;


%figure, imshow(sigma_map_BM4D(:,:,1),[]);colormap(gca,jet);colorbar;
% ************ Aplying Inverse Transform ******************** %

I_denoised = riceVST_EUI(I_VST_denoised, 1,'B');
I_denoised = I_denoised.*Estimated_NoiseMap;
%I_denoised = sqrt(I_VST_denoised.^2 + 16);
%******* Multiplying denoised image by the Estimated map ****%
else

max_I_noisy = max(I_noisy(:));
min_I_noisy = min(I_noisy(:));

I_noisy_norm = (I_noisy - min_I_noisy)/(max_I_noisy - min_I_noisy); 

scale_range = 0.7; scale_shift = 0.15;
%figure, imshow(I_noisy(:,:,1),[])
I_noisy_norm = (I_noisy_norm*scale_range + scale_shift);
%figure, imshow(I_noisy_norm(:,:,1),[])
[I_denoised_norm, sigma_map_BM4D] = bm4d(I_noisy_norm,'Rice', 0, profile, 1, 0);

%figure, imshow(I_denoised(:,:,1),[])
I_denoised = ((I_denoised_norm) - scale_shift)/scale_range;
I_denoised = I_denoised*(max_I_noisy - min_I_noisy) + min_I_noisy;
%figure, imshow(sigma_map_BM4D(:,:,1),[]);colormap(gca,jet);colorbar
sigma_map_BM4D = sigma_map_BM4D/scale_range;
sigma_map_BM4D = sigma_map_BM4D*(max_I_noisy - min_I_noisy);
%figure, imshow(sigma_map_BM4D(:,:,5),[]);colormap(gca,jet);colorbar

end



% figure, histogram(I_VST_denoised(:))
% figure, histogram(I_denoised(:))


end
