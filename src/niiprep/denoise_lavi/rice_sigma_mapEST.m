
% Iterative estimation of the rice_sigmaMap

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
%      T. Pieciak, S. Aja-Fernandez, G. Vegas-Sanches-Ferrero, 
%      Non-Stationary Rician Noise Estimation in Parallel MRI Using a Single Image: A Variance-Stabilizing Approach, 
%      IEEE Transactions on Pattern Analysis and Machine Intelligence, DOI: 10.1109/TPAMI.2016.2625789
% 
%
%      Spatially variant noise estimation in MRI: A homomorphic approach
%      S Aja-Fernández, T Pieciak, G Vegas-Sánchez-Ferrero
%      Medical Image Analysis, 2014
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Coding implement based on the previous algorithms of the papers cited
% above
%
% This work should only be used for nonprofit purposes.
%
% AUTHOR:
%     Vinícius P. Campos, email: vinicius.campos@usp.br
%     Updated - Nov/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Estimated_NoiseMap] = rice_sigma_mapEST(I_noisy,lpf_sigma)

if(~exist('lpf_sigma', 'var'))
    lpf_sigma = 3.4;
end

%initial estimation - no stabilization - homomorphic approach
[Estimated_NoiseMap] = homomorphic_gauss_estimation(I_noisy, lpf_sigma);
    

%Approach based on iterative searching the map - using FOI. VST

j=0;
converged=0;
maxIter = 8;

r_tolerance=0.0001;


while (j<maxIter)&&(converged==0)
    j=j+1;
    Estimated_NoiseMap_old=Estimated_NoiseMap;
    
    %Estimate after stabilization
    data_vst = Estimated_NoiseMap .* riceVST(I_noisy./Estimated_NoiseMap,1,'B');
 

    Estimated_NoiseMap = homomorphic_gauss_estimation(data_vst,lpf_sigma);


    rel_delta_matrix = ((Estimated_NoiseMap_old-Estimated_NoiseMap).^2)./((Estimated_NoiseMap_old).^2);


    rel_delta_max = sqrt(mean(rel_delta_matrix(:)));
        
    if rel_delta_max<r_tolerance %convergence check
        converged=1;
    end
    
   
end
     
end




