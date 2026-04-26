
% Iterative estimation of the rice_sigmaMap

function [Estimated_NoiseMap, rel_delta_vector] = rice_sigma_mapEST(I_noisy,estimation_type, lpf_sigma)

if(~exist('estimation_type', 'var'))
    estimation_type = 0;
end

if(~exist('wavelet_k', 'var'))
    wavelet_k = 'db7';
end

%initial estimation - no stabilization - homomorphic approach on a RICIAN
%data
if size(I_noisy,3)>1
    [Estimated_NoiseMap] = homomorphic_gauss_estimation(I_noisy, lpf_sigma, 'db7',4);
    
else
    %[Estimated_NoiseMap, ~] = rice_homomorf_est(I_noisy, 0,lpf_sigma,5,'db7');
     [Estimated_NoiseMap] = homomorphic_gauss_estimation(I_noisy, lpf_sigma, 'db7',4);
    
    
end

%figure, imshow(Estimated_NoiseMap,[]);colormap(gca,jet);

%Verify closeness to stationary

%sigma_estimated = riceVST_sigmaEst(I_noisy,'B');
%non_stat_map = ones(size(I_noisy))*sigma_estimated;

%rician_sigma_est = sigma_estimated;

%figure, imshow([Estimated_NoiseMap(:,:,1) non_stat_map(:,:,1)],[]);colormap(gca,jet);

%error_maps = mean2(sqrt(local_variance_estimate(Estimated_NoiseMap,9)));
%check error between maps
%error_maps = sqrt( mean2( ( (Estimated_NoiseMap - non_stat_map).^2 ) ) )./sigma_estimated;

%error_maps = mean2(abs(diff((Estimated_NoiseMap))));


converged=0;
error_maps = 10;
if abs(error_maps)<0.09 %if very close to stationary, returns a constant map
    %   Estimated_NoiseMap = non_stat_map;
    %   data_vst =   riceVST(I_noisy./Estimated_NoiseMap,1,'B')  ;
else
    
    %Estimate variant map
    
    if estimation_type==1  %Approach based on Pieciak pipeline
        rel_delta_vector = 0;
        
        data_vst = Estimated_NoiseMap .*riceVST_PIECAK_old(I_noisy,Estimated_NoiseMap);
        
        %Estimate map based on a Gaussian Homomorphic approach
        noise_extraction_method_PIECAK = 4;
        
        Estimated_NoiseMap = homomorphic_gauss_estimation(data_vst, lpf_sigma, wavelet_k, noise_extraction_method_PIECAK);
        
        
    else
        %Approach based on iterative searching the map - using FOI. VST
        
        j=0;
        
        maxIter = 20;
        
        r_tolerance=0.0005;
        
        %Estimated_NoiseMap = repmat(mean(Estimated_NoiseMap,3),[1 1 size(Estimated_NoiseMap,3)]);
        
        while (j<maxIter)&&(converged==0)
            j=j+1;
            Estimated_NoiseMap_old=Estimated_NoiseMap;
            %Apply VST
            
            sigma = 1 ; % after division, it turns to stationary with sigma=1
            
            %Estimate map based on a Gaussian Homomorphic approach
            data_vst = Estimated_NoiseMap .* riceVST(I_noisy./Estimated_NoiseMap,sigma,'B');
            noise_extraction_method = 4;
            
            
            Estimated_NoiseMap = homomorphic_gauss_estimation(data_vst,lpf_sigma, wavelet_k,noise_extraction_method);
            
            
            rel_delta_matriz = ((Estimated_NoiseMap_old-Estimated_NoiseMap).^2)./((Estimated_NoiseMap_old).^2);
            
            for i=1:size(rel_delta_matriz,3)
                rel_delta_vector(i) = sqrt(mean2(rel_delta_matriz(:,:,i)));
            end
            
            rel_delta_max = max(rel_delta_vector);
            
            if rel_delta_max<r_tolerance %convergence check
                converged=1;
            end
            
        end
        
    end
    
end




