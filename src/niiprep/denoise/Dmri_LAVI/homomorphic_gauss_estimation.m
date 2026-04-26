function [std_variant_map] = homomorphic_gauss_estimation(data,lpf_sigma, wavelet_k,noise_extraction_method)

if(~exist('wavelet_k', 'var'))
    wavelet_k = 'db7';
end

% Euler-Mascheroni constant
EulerMascheroni = 0.5772156649015328606065120;

% noise extraction from variance-stabilized Rician distributed data
% wavelet operator
if(noise_extraction_method == 1) 
    
    for i =1:size(data,3)
        [~, ~, ~, cD] = swt2(data(:,:,i) , 1, wavelet_k);
        noise_component(:,:,i) = abs(cD);      
    end
    

% bilateral filter	
elseif(noise_extraction_method == 2) 
    M = bilateral(data, local_window, bf_sigma_r, bf_sigma_g);
    noise_component = abs(data - M);

% local mean
elseif(noise_extraction_method == 3) 
    M = filter2B(ones(5)./prod(5), data);
    noise_component = abs(data - M);
    
elseif(noise_extraction_method == 4)  %proposed - based on wavelet transform
    scalar = 0;
   
    noise_component = abs(function_stdEst(data,[4 1],scalar,-1,[1 2]));    

end

data_log_transformed = log(noise_component.*(noise_component ~= 0) + 0.001.*(noise_component == 0));

data_log_transformed_lp = lpf(data_log_transformed, lpf_sigma, 1);

std_variant_map = sqrt(2) .* exp(data_log_transformed_lp + EulerMascheroni./2);
  

if size(data_log_transformed,3)>1
    sigma_xy = 1.8;
    sigma_z = 1.5;
    sigma_vector = [sigma_xy sigma_xy sigma_z];
    
    wind_siz_xy = 3;
    wind_siz_z = 5;
    wind_siz_vector = [wind_siz_xy wind_siz_xy wind_siz_z];
 
    std_variant_map= imgaussfilt3(std_variant_map,sigma_vector,'FilterSize',wind_siz_vector,'padding','symmetric','FilterDomain','frequency');
    
    
end

end

