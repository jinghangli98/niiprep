%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Please cite the following papers:
%
%      Campos, V.P. , Santini, T. , Borges, L.R. , Ibrahim, T. S.  and
%      Vieira, MAC "A denoising framework based on variance stabilization 
%      for volumetric magnetic resonance images corrupted with 
%      non-stationary Rician noise", *********
%
%
%      T. Pieciak, S. Aja-Fernandez, G. Vegas-Sanches-Ferrero, 
%      Non-Stationary Rician Noise Estimation in Parallel MRI Using a Single Image: A Variance-Stabilizing Approach, 
%      IEEE Transactions on Pattern Analysis and Machine Intelligence, DOI: 10.1109/TPAMI.2016.2625789
% 
%
%      Spatially variant noise estimation in MRI: A homomorphic approach
%      S Aja-Fernández, T Pieciak, G Vegas-Sánchez-Ferrero
%      Medical Image Analysis, 2014

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Coding implemented based on the previous algorithms provided by the
% authors/papers cited above
%
% This work should only be used for nonprofit purposes.
%
% AUTHOR:
%     Vinícius P. Campos, email: vinicius.campos@usp.br
%     Updated - Aug/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [std_variant_map] = homomorphic_gauss_estimation(data,lpf_sigma)

% Euler-Mascheroni constant
EulerMascheroni = 0.5772156649015328606065120;

% noise extraction using wavelet function
scalar=0;

wavlt = 3;

noise_component1 = abs(function_stdEst(data,[wavlt 1],scalar,1,[1 2]));    

  
noise_component2 = abs(function_stdEst(permute(data,[1 3 2]),[wavlt 1],scalar,1,[1 2]));    
noise_component2 = permute(noise_component2,[1 3 2]);
   
noise_component3 = abs(function_stdEst(permute(data,[3 2 1]),[wavlt 1],scalar,1,[1 2]));    
noise_component3 = permute(noise_component3,[3 2 1]);


% log of abs value
data_log_transformed1 = log(noise_component1.*(noise_component1 ~= 0) + 0.00001.*(noise_component1 == 0));
data_log_transformed2 = log(noise_component2.*(noise_component2 ~= 0) + 0.00001.*(noise_component2 == 0));
data_log_transformed3 = log(noise_component3.*(noise_component3 ~= 0) + 0.00001.*(noise_component3 == 0));

%data_log_transformed = log(noise_component.*(noise_component ~= 0) + 0.00001.*(noise_component == 0));

%low-pass filter (lpf)
data_log_transformed_lp1 = lpf(data_log_transformed1, lpf_sigma, 1);
data_log_transformed_lp2 = lpf(data_log_transformed2, lpf_sigma, 1);
data_log_transformed_lp3 = lpf(data_log_transformed3, lpf_sigma, 1);



%apply exponentional to obtain std map
std_variant_map1 = sqrt(2) .* exp(data_log_transformed_lp1 + EulerMascheroni./2);
std_variant_map2 = sqrt(2) .* exp(data_log_transformed_lp2 + EulerMascheroni./2);
std_variant_map3 = sqrt(2) .* exp(data_log_transformed_lp3 + EulerMascheroni./2);


std_variant_map = (std_variant_map1 + std_variant_map2 + std_variant_map3)/3;



%Gaussian 3d conv kernel to enhace each 2D estimate
if size(std_variant_map,3)>1
    sigma_xy = 5;
    sigma_z = 5;
    sigma_vector = [sigma_xy sigma_xy sigma_z];
    
    wind_siz_xy = 11;
    wind_siz_z = 11;
    wind_siz_vector = [wind_siz_xy wind_siz_xy wind_siz_z];
    
    
    std_variant_map= imgaussfilt3(std_variant_map,sigma_vector,'FilterSize',wind_siz_vector,'padding','symmetric','FilterDomain','spatial');
      
    
end

end
