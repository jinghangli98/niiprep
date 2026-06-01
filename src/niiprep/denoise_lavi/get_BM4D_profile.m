
function [BM4D_profile]= get_BM4D_profile(profile)

if(~exist('profile', 'var'))
    profile = 'np';
end


BM4D_profile = BM4DProfile('np');


switch profile

    case 'lc'
        %fprintf('\nBM4D_profile = %s \n',profile)

        BM4D_profile.N1               = [4 4 4];
        BM4D_profile.N2               = 16;
        BM4D_profile.Ns               = [3 3 3];

        BM4D_profile.N1_wiener        = [4 4 4];
        BM4D_profile.N2_wiener        = 16;
        BM4D_profile.Ns_wiener        = [3 3 3];


    case 'np'
        %fprintf('\nBM4D_profile = %s \n',profile)

        BM4D_profile.N1               = [4 4 4];
        BM4D_profile.N2               = 16;
        BM4D_profile.Ns               = [5 5 5];

        BM4D_profile.N1_wiener        = [4 4 4];
        BM4D_profile.N2_wiener        = 32;
        BM4D_profile.Ns_wiener        = [5 5 5];


    case 'mp'
        %fprintf('\nBM4D_profile = %s \n',profile)

        BM4D_profile.N1               = [4 4 4];
        BM4D_profile.N2               = 32;
        BM4D_profile.Ns               = [5 5 5];

        if ispc
            BM4D_profile.N1_wiener        = [4 4 4]; %there is currently a bug on windows version. So we need to maintain the block size = [4 4 4]
        else
            BM4D_profile.N1_wiener        = [5 5 5];
        end

        BM4D_profile.N2_wiener        = 32;
        BM4D_profile.Ns_wiener        = [5 5 5];


end

end
