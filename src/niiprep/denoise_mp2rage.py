import os
import numpy as np
import nibabel as nib
import json
from matplotlib import pyplot as plt
from scipy import ndimage

try:
    from .utils import load_dicom_with_sitk, save_enhanced_dicom
except ImportError:
    from niiprep.utils import load_dicom_with_sitk, save_enhanced_dicom

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
    if isinstance(regularization, str) and regularization.lower() == 'auto':
        auto_reg = True
        multiplying_factor = 1.0
    else:
        auto_reg = False
        multiplying_factor = 1 if regularization is None else regularization

    final_choice = 'n'
    
    # Define helper functions
    def mp2rage_robust_func(inv1, inv2, beta):
        return (np.conj(inv1) * inv2 - beta) / (inv1**2 + inv2**2 + 2*beta + 1e-5)
    
    def roots_pos(a, b, c):
        return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a + 1e-5)
    
    def roots_neg(a, b, c):
        return (-b - np.sqrt(b**2 - 4*a*c)) / (2*a + 1e-5)
    
    def fill_nan_voxels(array, radius=1):
        """Fill NaN voxels with mean of non-NaN neighbors."""
        nan_mask = np.isnan(array)
        
        if not np.any(nan_mask):
            return array
        
        print(f"Found {np.sum(nan_mask)} NaN voxels, filling with neighborhood mean...")
        
        def mean_of_valid(neighborhood):
            valid = neighborhood[~np.isnan(neighborhood)]
            return np.mean(valid) if len(valid) > 0 else 0
        
        filled = ndimage.generic_filter(array, mean_of_valid, size=2*radius+1)
        array[nan_mask] = filled[nan_mask]

        return array

    def detect_optimal_regularization(inv1_final, inv2_data):
        """Auto-pick the regularization multiplier via a noise-floor knee.

        Sweeps integer multipliers, measures how clean the air/background gets
        for each, and stops at the point where adding more regularization no
        longer meaningfully reduces background noise (the "knee").
        """
        print("Auto-detecting regularization factor...")
        # Use central slice
        z_idx = inv1_final.shape[2] // 2
        inv1_slice = inv1_final[:, :, z_idx]
        inv2_slice = inv2_data[:, :, z_idx]

        # Background mask (corner 10%)
        h, w = inv1_slice.shape
        bg_mask = np.zeros((h, w), dtype=bool)
        bg_mask[:int(h*0.1), :int(w*0.1)] = True
        bg_mask[-int(h*0.1):, -int(w*0.1):] = True

        # Base noise from slice corner (matches the full-volume loop below)
        base_noise = np.mean(inv2_slice[-10:, -10:])

        factors = list(range(1, 16))
        stds = []

        for f in factors:
            noise_level = f * base_noise
            beta = noise_level**2
            num = np.conj(inv1_slice) * inv2_slice - beta
            den = inv1_slice**2 + inv2_slice**2 + 2*beta + 1e-5
            robust = num / den
            stds.append(np.std(robust[bg_mask]))

        # Knee detection: stop once extra regularization stops cleaning the
        # background (noise reduction drops below 5% of the largest reduction).
        stds = np.array(stds)
        diffs = -np.diff(stds)  # reduction in background noise per +1 factor

        max_red = np.max(diffs) if len(diffs) > 0 else 0
        threshold = 0.05 * max_red

        optimal = factors[-1]
        for i in range(len(diffs)):
            if diffs[i] < threshold:
                optimal = factors[i]
                break

        print(f"Detected optimal regularization: {optimal}")
        return float(optimal)

    def load_image(path):
        """Load a NIfTI file or a DICOM-series directory -> (data, affine)."""
        if os.path.isdir(path):
            print(f"Loading DICOM from: {path}")
            _, data, affine = load_dicom_with_sitk(path, reorient=True)
            return data.astype(np.float64), affine
        elif os.path.exists(path):
            print(f"Loading NIfTI from: {path}")
            img = nib.load(path)
            return img.get_fdata().astype(np.float64), img.affine
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    # Load data
    uni_data, uni_affine = load_image(mp2rage['filenameUNI'])
    inv1_data, _ = load_image(mp2rage['filenameINV1'])
    inv2_data, _ = load_image(mp2rage['filenameINV2'])
    
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

    if auto_reg:
        multiplying_factor = detect_optimal_regularization(inv1_final, inv2_data)
        regularization = multiplying_factor

    # Interactive regularization loop
    while final_choice.lower() != 'y':
        noise_level = multiplying_factor * np.mean(inv2_data[:, -10:, -10:])
        mp2rage_robust = mp2rage_robust_func(inv1_final, inv2_data, noise_level**2)
        
        # Fill NaN voxels with neighborhood mean
        mp2rage_robust = fill_nan_voxels(mp2rage_robust, radius=1)
        
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
        output_path = mp2rage['filenameOUT']
        print(f"Saving: {output_path}")
        out_data = mp2rage_robust if not integer_format else np.round(4095 * (mp2rage_robust + 0.5))

        is_nifti = output_path.endswith('.nii') or output_path.endswith('.nii.gz')

        if is_nifti:
            out_img = nib.Nifti1Image(out_data, uni_affine)
            nib.save(out_img, output_path)

            # Handle JSON sidecar (only when the UNI input is a file with a sidecar)
            if os.path.isfile(mp2rage['filenameUNI']):
                uni_json = os.path.splitext(mp2rage['filenameUNI'])[0] + '.json'
                if os.path.exists(uni_json):
                    with open(uni_json, 'r') as f:
                        json_data = json.load(f)

                    json_data.update({
                        'BasedOn': [mp2rage['filenameUNI'], mp2rage['filenameINV1'], mp2rage['filenameINV2']],
                        'SeriesDescription': f"{json_data.get('ProtocolName', 'Unknown')}_MP2RAGE_denoised_background",
                        'NoiseRegularization': multiplying_factor
                    })

                    out_json = output_path.split('.nii')[0] + '.json'
                    with open(out_json, 'w') as f:
                        json.dump(json_data, f)
        else:
            # DICOM output: requires the UNI input to be a DICOM-series directory
            # to use as the metadata template.
            if os.path.isdir(mp2rage['filenameUNI']):
                spacing = nib.affines.voxel_sizes(uni_affine)
                dicom_data = np.clip(out_data, 0, 4095)
                save_enhanced_dicom(dicom_data, mp2rage['filenameUNI'], output_path,
                                    new_spacing=spacing,
                                    series_description_suffix="_MP2RAGE_denoised_background")
            else:
                raise ValueError(
                    "DICOM output requested (non-.nii output path) but the UNI input "
                    "is not a DICOM-series directory; cannot build DICOM metadata."
                )

    return mp2rage_robust, multiplying_factor

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MP2RAGE robust combination processing')
    
    parser.add_argument('--uni', type=str, required=True,
                        help='Path to UNI image (.nii/.nii.gz file or DICOM-series directory)')
    parser.add_argument('--inv1', type=str, required=True,
                        help='Path to INV1 image (.nii/.nii.gz file or DICOM-series directory)')
    parser.add_argument('--inv2', type=str, required=True,
                        help='Path to INV2 image (.nii/.nii.gz file or DICOM-series directory)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path (.nii/.nii.gz file, or a directory for DICOM output)')

    def parse_reg(value):
        if value.lower() == 'auto':
            return 'auto'
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value for regularization: {value}")

    parser.add_argument('--regularization', '-r', type=parse_reg, default='auto',
                        help="Noise regularization factor; a number, or 'auto' to "
                             "auto-detect (default: 'auto')")

    args = parser.parse_args()
    mp2rage_data = {
        'filenameUNI': args.uni,
        'filenameINV1': args.inv1,
        'filenameINV2': args.inv2,
        'filenameOUT': args.output
    }
    robust_image, factor = robust_combination(mp2rage_data, regularization=args.regularization,)
