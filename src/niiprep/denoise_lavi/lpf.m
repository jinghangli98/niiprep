function If=lpf(I,sigma,MODO)
%
% LPF low pass filter of images 
%       If=lpf(I,sigma,MODO)
% 
% INPUT:
%   	I:	Input Image
%		sigma:	standard deviation of Gaussian window in 
%               frequancy domain (related to filter bandwidth)
%		MODO:   1: DFT filtering
%               2: DCT filtering
%
% Santiago Aja-Fernandez (V1.0)
% LPI 
% www.lpi.tel.uva.es/~santi
% sanaja@tel.uva.es
% LPI Valladolid, Spain
% Original: 06/07/2014, 
% Release   16/12/2014
%
% RICE HOMOMORPHIC TOOLBOX
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Original code is from the above authors/papers.
%
%     Included minor changes: 
%     1 - output is the abs of the ifft, instead of real
%
%     2 - 3D data considered (elseif clause for the DFT case (MODO=1) )
%         Used repmat(). This way, each 2D is filtered separately
%         This easily computes 2D filtering for all slices on the volume
%     
%     Vinícius P. Campos, email: vinicius.campos@usp.br
%     Updated - Aug/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(~exist('MODO','var'))
    MODO=1;
end

%MODO==1 DFT
if MODO==1
    [Mx,My,Mz]=size(I);
    h=fspecial('gaussian',[Mx My],sigma);
  
    h=h./max(h(:));
      
   if (Mx==1)||(My==1) %1D
       lRnF=fftshift(fft(I));
       %Filtering
       lRnF2=lRnF.*h;
       If=abs(ifft(fftshift(lRnF2)));
      
        
   elseif size(I,3)>1 %3D data. 
       lRnF=fftshift(fft2(I));
       %Filtering
       h = repmat(h,[1 1 Mz]);
           
       lRnF2=lRnF.*h;
       If=abs(ifft2(fftshift(lRnF2)));
        
        
    else %2D
        lRnF=fftshift(fft2(I));
        %Filtering
        lRnF2=lRnF.*h;
        If=abs(ifft2(fftshift(lRnF2)));
    end
    
  
%MODO==2 DCT
elseif MODO==2
    [Mx,My]=size(I);
    h=fspecial('gaussian',2.*[Mx,My],sigma.*2);
    h=(1)*h./max(h(:));
    h=h((Mx+1):end,(My+1):end);

    if (Mx==1)||(My==1) %1D
        lRnF=dct(I);
        %Filtering
        lRnF2=lRnF.*h;
        If=real(idct(lRnF2));

    else %2D
        lRnF=dct2(I);
        %Filtering
        lRnF2=lRnF.*h;
        If=real(idct2(lRnF2));
    end
end
