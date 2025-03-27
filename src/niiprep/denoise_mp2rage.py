import os
import numpy as np
import nibabel as nib
import json
from matplotlib import pyplot as plt

def robust_combination(mp2rage, regularization=None, figure=None):
    """
    Creates MP2RAGE T1w images without strong background noise in air regions.
    This python file is translated from 
    
    https://github.com/JosePMarques/MP2RAGE-related-scripts/tree/master

    Parameters:
    -----------
    mp2rage : dict
        Dictionary containing filenames:
        - filenameUNI: Path to UNI image
        - filenameINV1: Path to INV1 image
        - filenameINV2: Path to INV2 image
        - filenameOUT: Optional output path
    regularization : float, optional
        Noise regularization factor
    figure : matplotlib.figure, optional
        Figure handle for visualization
    
    Returns:
    --------
    tuple: (mp2rage_robust, multiplying_factor)
    """
    
    # Set defaults
    multiplying_factor = 1 if regularization is None else regularization
    final_choice = 'n'
    
    # Define helper functions
    def mp2rage_robust_func(inv1, inv2, beta):
        return (np.conj(inv1) * inv2 - beta) / (inv1**2 + inv2**2 + 2*beta)
    
    def roots_pos(a, b, c):
        return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    def roots_neg(a, b, c):
        return (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    # Load data
    print(f"Loading images from: {os.path.dirname(mp2rage['filenameUNI'])}")
    uni_img = nib.load(mp2rage['filenameUNI'])
    uni_data = uni_img.get_fdata().astype(np.float64)
    inv1_data = nib.load(mp2rage['filenameINV1']).get_fdata().astype(np.float64)
    inv2_data = nib.load(mp2rage['filenameINV2']).get_fdata().astype(np.float64)
    
    # Check if normalization is needed
    if uni_data.min() >= 0 and uni_data.max() >= 0.51:
        uni_data = (uni_data - uni_data.max()/2) / uni_data.max()
        integer_format = True
    else:
        integer_format = False
    
    # Compute correct INV1 dataset
    inv1_data = np.sign(uni_data) * inv1_data
    
    inv1_pos = roots_pos(-uni_data, inv2_data, -inv2_data**2 * uni_data)
    inv1_neg = roots_neg(-uni_data, inv2_data, -inv2_data**2 * uni_data)
    
    inv1_final = inv1_data.copy()
    mask_neg = np.abs(inv1_data - inv1_pos) > np.abs(inv1_data - inv1_neg)
    inv1_final[mask_neg] = inv1_neg[mask_neg]
    inv1_final[~mask_neg] = inv1_pos[~mask_neg]
    
    # Interactive regularization loop
    while final_choice.lower() != 'y':
        noise_level = multiplying_factor * np.mean(inv2_data[:, -10:, -10:])
        mp2rage_robust = mp2rage_robust_func(inv1_final, inv2_data, noise_level**2)
        
        if figure is not None:
            # Visualization code here (simplified)
            plt.figure(figure.number)
            plt.subplot(211)
            plt.imshow(uni_data[:, :, uni_data.shape[2]//2], cmap='gray', vmin=-0.5, vmax=0.4)
            plt.title('MP2RAGE UNI-Image')
            
            plt.subplot(212)
            plt.imshow(mp2rage_robust[:, :, mp2rage_robust.shape[2]//2], cmap='gray', vmin=-0.5, vmax=0.4)
            plt.title(f'MP2RAGE Robust (Noise level = {multiplying_factor})')
            plt.show()
            
            if regularization is None:
                final_choice = input('Is it a satisfactory noise level?? (y/n) [n]: ') or 'n'
                if final_choice.lower() != 'y':
                    multiplying_factor = float(input(f'New regularization noise level (current = {multiplying_factor}): '))
            else:
                final_choice = 'y'
        else:
            final_choice = 'y'
    
    # Save output if filename provided
    if 'filenameOUT' in mp2rage and mp2rage['filenameOUT']:
        print(f"Saving: {mp2rage['filenameOUT']}")
        out_data = mp2rage_robust if not integer_format else np.round(4095 * (mp2rage_robust + 0.5))
        out_img = nib.Nifti1Image(out_data, uni_img.affine)
        nib.save(out_img, mp2rage['filenameOUT'])
        
        # Handle JSON sidecar
        uni_json = os.path.splitext(mp2rage['filenameUNI'])[0] + '.json'
        if os.path.exists(uni_json):
            with open(uni_json, 'r') as f:
                json_data = json.load(f)
            
            json_data.update({
                'BasedOn': [mp2rage['filenameUNI'], mp2rage['filenameINV1'], mp2rage['filenameINV2']],
                'SeriesDescription': f"{json_data['ProtocolName']}_MP2RAGE_denoised_background",
                'NoiseRegularization': multiplying_factor
            })
            
            out_json = os.path.splitext(mp2rage['filenameOUT'])[0] + '.json'
            with open(out_json, 'w') as f:
                json.dump(json_data, f)
    
    return mp2rage_robust, multiplying_factor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MP2RAGE robust combination processing')
    
    parser.add_argument('--uni', type=str, required=True,
                        help='Path to UNI image (.nii or .nii.gz)')
    parser.add_argument('--inv1', type=str, required=True,
                        help='Path to INV1 image (.nii or .nii.gz)')
    parser.add_argument('--inv2', type=str, required=True,
                        help='Path to INV2 image (.nii or .nii.gz)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path for processed image')
    parser.add_argument('--regularization', '-r', type=float, default=None,
                        help='Noise regularization factor (default: None for interactive mode)')

    args = parser.parse_args()
    mp2rage_data = {
        'filenameUNI': args.uni,
        'filenameINV1': args.inv1,
        'filenameINV2': args.inv2,
        'filenameOUT': args.output
    }
    robust_image, factor = robust_combination(mp2rage_data, regularization=args.regularization,)