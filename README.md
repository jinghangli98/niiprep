# NiiPrep

A CLI wrapper for TorchIO and ANTsPyX for NIfTI image processing

## Overview

NiiPrep is a Python package that provides convenient command-line tools for common neuroimaging preprocessing tasks. It combines the power of TorchIO and ANTsPyX libraries to offer streamlined workflows for NIfTI image manipulation, registration, resampling, and visualization.

## Features

- **Image Resampling**: Change voxel spacing with multiple interpolation methods
- **Image Registration**: Rigid, affine, and deformable registration using ANTsPyX
- **Image Denoising**: Non-local means denoising using ANTs, plus MATLAB-based Rice/VST denoising
- **Bias Field Correction**: N4 bias field correction using ANTs, or SPM12-based bias correction
- **Video Conversion**: Convert NIfTI files to MP4 videos for visualization
- **Image Cropping/Padding**: Resize images to specified dimensions or automatically remove empty space
- **Value Rounding**: Round pixel values in NIfTI images
- **MP2RAGE Denoising**: Robust combination processing for MP2RAGE images
- **3D Patch Parcellation**: Split NIfTI volumes into 3D patches and reconstruct, with automatic background filtering
- **Maximum Intensity Projection**: Full or sliding-slab MIPs and rotating MIP cines (GIF/MP4) for TOF MRA vascular rendering, with optional GPU acceleration

## Installation

```bash
pip install niiprep
```

### Requirements

- Python >= 3.7
- TorchIO >= 0.18.0
- ANTsPyX >= 0.3.0
- NiBabel >= 3.0.0
- NumPy >= 1.19.0
- OpenCV (for video conversion)
- Matplotlib (for MP2RAGE processing)
- patchify >= 0.1.0 (for patch parcellation)
- MATLAB (for `mdenoise`)
- MATLAB + SPM12 (for `mbiascorrect`)

## Command-Line Tools

After installation, the following commands are available:

### 1. `resample` - Image Resampling

Resample NIfTI images to specified voxel spacing.

```bash
resample -i input.nii.gz -o output.nii.gz -s 1.0 1.0 1.0 --interpolation linear
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save resampled NIfTI file
- `-s, --spacing`: Target voxel spacing in mm (x y z), default: 1.0 1.0 1.0
- `--interpolation`: Interpolation method (linear, nearest, bspline), default: linear

### 2. `registernii` - Image Registration

Register moving image to fixed image using ANTsPyX.

```bash
registernii -f fixed.nii.gz -m moving.nii.gz -o registered.nii.gz -t syn --interpolation linear
registernii -f fixed.nii.gz -m moving.nii.gz -o registered.nii.gz -t syn --norm
```

**Parameters:**
- `-f, --fixed`: Path to fixed/reference NIfTI file
- `-m, --moving`: Path to moving NIfTI file
- `-o, --output`: Path to save registered NIfTI file
- `-t, --type`: Registration type (rigid, affine, syn), default: syn
- `--interpolation`: Interpolation type, default: linear
- `--norm`: Min-max normalize intensities to 0-255 and round before saving (optional)

### 3. `nii2mp4` - Video Conversion

Convert NIfTI files to MP4 videos for visualization.

```bash
nii2mp4 -i input.nii.gz -o output.mp4 -d 2 --fps 10
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save MP4 file
- `-d, --dimension`: Dimension to slice along (0: sagittal, 1: coronal, 2: axial), default: 2
- `--fps`: Frames per second, default: 10
- `--no-normalize`: Disable intensity normalization

### 4. `crop` - Image Cropping/Padding

Crop or pad NIfTI images to specified shape.

```bash
crop -i input.nii.gz -o output.nii.gz -s 256 256 256
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save cropped/padded NIfTI file
- `-s, --shape`: Target image shape, default: 256 256 256

### 5. `roundnii` - Value Rounding

Round pixel values in NIfTI images.

```bash
roundnii -i input.nii.gz
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file (will be overwritten)

### 6. `denoise` - Image Denoising

Denoise NIfTI images using ANTs denoising algorithm. Values are rounded before saving, and can optionally be normalized to 0-255 range.

```bash
denoise -i input.nii.gz -o denoised.nii.gz
denoise -i input.nii.gz -o denoised.nii.gz --norm
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save denoised NIfTI file
- `--norm`: Min-max normalize intensities to 0-255 and round before saving (optional)

### 7. `mdenoise` - MATLAB Image Denoising

Denoise NIfTI images using the bundled MATLAB Rice/VST denoising routine. Requires MATLAB to be installed and available on the system `PATH`.

```bash
mdenoise -i input.nii.gz -o denoised.nii.gz
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save denoised NIfTI file

**Notes:**
- The command converts the input to a temporary `.nii` file for MATLAB, runs `new_denoising_func_matlab_matrix`, and saves the first denoised output.
- MATLAB assets are packaged under `niiprep/denoise`.

### 8. `biascorrect` - N4 Bias Field Correction

Apply N4 bias field correction using ANTs. Values are rounded before saving, and can optionally be normalized to 0-255 range.

```bash
biascorrect -i input.nii.gz -o corrected.nii.gz
biascorrect -i input.nii.gz -o corrected.nii.gz --norm
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save bias-corrected NIfTI file
- `--norm`: Min-max normalize intensities to 0-255 and round before saving (optional)

### 9. `denoiseMP2RAGE` - MP2RAGE Denoising

Process MP2RAGE images with robust combination to reduce background noise.

```bash
denoiseMP2RAGE --uni uni.nii.gz --inv1 inv1.nii.gz --inv2 inv2.nii.gz -o denoised.nii.gz -r 1.0
```

**Parameters:**
- `--uni`: Path to UNI image
- `--inv1`: Path to INV1 image
- `--inv2`: Path to INV2 image
- `-o, --output`: Output path for processed image
- `-r, --regularization`: Noise regularization factor (optional, interactive mode if not specified)

### 10. `mbiascorrect` - SPM12 Bias Field Correction

Apply SPM12 unified segmentation-based bias field correction. Accepts `.nii` or `.nii.gz` inputs. Requires MATLAB and SPM12 to be installed and on the system `PATH`.

```bash
mbiascorrect -i input.nii.gz -o corrected.nii.gz
mbiascorrect -i input.nii.gz -o corrected.nii.gz --save-masks
mbiascorrect -i input.nii.gz -o corrected.nii.gz --spm-path /path/to/spm12
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file (`.nii` or `.nii.gz`)
- `-o, --output`: Path to save bias-corrected NIfTI file (`.nii` or `.nii.gz`)
- `--spm-path`: Path to SPM12 installation (default: `/ihome/tibrahim/jil202/spm12`)
- `--window-size`: Bias FWHM in mm, controls smoothness of estimated bias field (default: 60)
- `--sphere-size`: Sphere size for brain mask in mm (default: 50)
- `--include-c3`: Include CSF (c3) tissue class in segmentation (optional)
- `--save-masks`: Save SPM tissue segmentation maps (`c1`–`c5`) alongside the output (optional)

**Notes:**
- SPM12 saves tissue probability maps for GM (`c1`), WM (`c2`), CSF (`c3`), bone (`c4`), and soft tissue (`c5`). These are discarded by default; use `--save-masks` to keep them.
- `.mat` segmentation parameter files generated by SPM are always discarded.

### 11. `autocropnii` - Automatic Image Cropping

Automatically crop NIfTI images to remove empty space (based on gradient detection) and optionally pad to a specific shape or multiple.

```bash
autocropnii -i input.nii.gz -o output.nii.gz -n 14
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save cropped/padded NIfTI file
- `-n`: Optional integer. If provided, output dimensions will be padded to be multiples of this value. Default is None (tight crop).
- `-s, --shape`: Optional target shape (x y z). If provided, output will maximize this shape. If used with `-n`, dimensions are adjusted to the closest smaller multiple of `n`.

### 12. `patchifynii` - 3D Patch Parcellation

Split a 3D NIfTI image into overlapping or non-overlapping patches. Each patch is saved as an individual `.nii.gz` file with a spatially correct affine, alongside a `patches_meta.json` used for reconstruction. Background and noise patches are automatically skipped using Otsu's threshold.

```bash
# Non-overlapping 32³ patches, auto background filtering
patchifynii -i input.nii.gz -o ./patches -p 32 32 32 --norm

# Overlapping patches (step < patch size)
patchifynii -i input.nii.gz -o ./patches -p 64 64 64 -s 32 --norm

# Stricter foreground filter (keep only patches with >20% foreground voxels)
patchifynii -i input.nii.gz -o ./patches -p 32 32 32 --norm --min-nonzero-frac 0.2
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Output directory for patches and metadata
- `-p, --patch-size`: Patch size in voxels (x y z)
- `-s, --step`: Step size between patches (default: `min(patch_size)`, non-overlapping)
- `--norm`: Normalize image intensities to [0, 1] before patchifying
- `--min-nonzero-frac`: Skip patches where fewer than this fraction of voxels are foreground (default: 0.01)
- `--min-intensity-range`: Skip patches whose max−min intensity ≤ this value (default: 0.0, disabled)
- `--foreground-threshold`: Intensity above which a voxel counts as foreground (default: auto Otsu on the full image)
- `--no-pad`: Raise an error instead of zero-padding when image dimensions are incompatible with patch/step size

**Notes:**
- Images are automatically zero-padded so dimensions satisfy `(dim − patch_size) % step == 0`. The original shape is stored in metadata and cropped back on reconstruction.
- The foreground threshold defaults to Otsu's threshold computed on the full image, so no manual tuning is needed.

### 13. `unpatchifynii` - Patch Reconstruction

Reconstruct a 3D NIfTI image from patches created by `patchifynii`. Skipped (background) patches are filled with zeros.

```bash
unpatchifynii -i ./patches -o reconstructed.nii.gz
```

**Parameters:**
- `-i, --input`: Directory containing patches and `patches_meta.json`
- `-o, --output`: Path to save reconstructed NIfTI file

### 14. `mip` - Maximum Intensity Projection

Compute a Maximum Intensity Projection (MIP) of a NIfTI volume, e.g. to render the vascular tree from a time-of-flight (TOF) MRA. Three modes are available:

- **Full projection** (default): collapse the whole volume along one axis into a single 2D projection (saved as a NIfTI with a singleton dimension).
- **Sliding-slab MIP** (`--slab`): for every slice, project through a centered slab of the given thickness. The output has the same shape as the input, so it can be scrolled like a regular volume.
- **Rotating cine** (`--gif` / `--mp4`): spin the volume 360° around `--spin-axis`, project along `--axis` at each angle, and write an animated GIF or MP4. Uses the GPU (CuPy) when available and falls back to the CPU otherwise.

```bash
# Axial MIP of a TOF angiogram (default axis 2)
mip -i tof.nii.gz -o tof_mip.nii.gz

# Sagittal / coronal projection
mip -i tof.nii.gz -o tof_mip_sag.nii.gz -a 0
mip -i tof.nii.gz -o tof_mip_cor.nii.gz -a 1

# Sliding-slab MIP, 20-voxel slab, same shape as the input
mip -i tof.nii.gz -o tof_slabmip.nii.gz --slab 20

# Rotating MIP cine as GIF (36 frames, 10 fps, spins around the cranio-caudal axis)
mip -i tof.nii.gz -o tof_spin.gif --gif -a 1

# Smoother MP4 cine, forced onto the CPU
mip -i tof.nii.gz -o tof_spin.mp4 --mp4 -a 1 --frames 72 --fps 15 --no-gpu
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save the output. A NIfTI file by default; with `--gif`/`--mp4` a `.nii`/`.nii.gz` suffix is replaced by the cine extension automatically
- `-a, --axis`: Axis to project along: `0` sagittal, `1` coronal, `2` axial (default: `2`). For a rotating cine, `-a 1` (coronal) with the default `--spin-axis 2` gives the classic spinning head view
- `--slab`: Slab thickness in voxels. Without a cine flag this produces a same-shape sliding-slab MIP volume; with `--gif`/`--mp4` each frame projects through a centered slab of this thickness (optional)
- `--gif`: Write a rotating MIP cine as an animated GIF instead of a NIfTI file
- `--mp4`: Write a rotating MIP cine as an MP4 video instead of a NIfTI file (mutually exclusive with `--gif`)
- `--spin-axis`: (`--gif`/`--mp4`) Axis the volume rotates around (default: `2`, cranio-caudal for TOF head data)
- `--frames`: (`--gif`/`--mp4`) Number of frames over a full 360° rotation (default: `36`)
- `--fps`: (`--gif`/`--mp4`) Playback frames per second (default: `10`)
- `--no-gpu`: (`--gif`/`--mp4`) Force the CPU (SciPy) backend even if a CUDA GPU is available

GPU acceleration for the rotating cine requires CuPy. Install it with the `gpu` extra or pick the wheel matching your CUDA toolkit:

```bash
pip install "niiprep[gpu]"
# or, for example
pip install cupy-cuda12x
```

## Python API

You can also use NiiPrep functions directly in Python:

```python
from niiprep import resample, register, nii_to_mp4, mdenoise
from niiprep.patchify_nii import patchify_nii, unpatchify_nii
from niiprep.mip import mip, rotating_mip

# Resample an image
resample(
    input_path='input.nii.gz',
    output_path='output.nii.gz',
    target_spacing=(1.0, 1.0, 1.0),
    interpolation='linear'
)

# Register images
register(
    fixed_path='fixed.nii.gz',
    moving_path='moving.nii.gz',
    output_path='registered.nii.gz',
    reg_type='syn'
)

# Convert to video
nii_to_mp4(
    input_path='input.nii.gz',
    output_path='output.mp4',
    dimension=2,
    fps=10
)

# MATLAB denoise
mdenoise(
    input_path='input.nii.gz',
    output_path='matlab_denoised.nii.gz',
)

# Split into 3D patches (auto background filtering, normalize to [0,1])
patchify_nii(
    input_path='input.nii.gz',
    output_dir='./patches',
    patch_size=(64, 64, 64),
    step=32,          # overlapping; omit for non-overlapping
    norm=True,
    min_nonzero_frac=0.1,
)

# Reconstruct from patches
unpatchify_nii(
    input_dir='./patches',
    output_path='reconstructed.nii.gz',
)

# Axial MIP of a TOF angiogram (slab=None -> single full projection)
mip(
    input_path='tof.nii.gz',
    output_path='tof_mip.nii.gz',
    axis=2,
    slab=None,
)

# Rotating MIP cine (coronal projection, spinning around the cranio-caudal axis)
rotating_mip(
    input_path='tof.nii.gz',
    output_path='tof_spin.mp4',
    axis=1,
    spin_axis=2,
    frames=72,
    fps=15,
    fmt='mp4',
    use_gpu=True,     # falls back to CPU if CuPy / a CUDA device is missing
)
```

## Examples

### Basic Preprocessing Pipeline

```bash
# 1. Denoise the raw image
denoise -i raw.nii.gz -o denoised.nii.gz

# 2. Apply bias field correction
biascorrect -i denoised.nii.gz -o corrected.nii.gz

# 3. Resample to 1mm isotropic
resample -i corrected.nii.gz -o resampled.nii.gz -s 1.0 1.0 1.0

# 4. Crop to standard size (or use autocrop for automatic sizing)
crop -i resampled.nii.gz -o cropped.nii.gz -s 256 256 256
# OR
autocropnii -i resampled.nii.gz -o cropped.nii.gz -n 14

# 5. Register to template with normalization
registernii -f template.nii.gz -m cropped.nii.gz -o registered.nii.gz -t affine --norm

# 6. Create visualization video
nii2mp4 -i registered.nii.gz -o preview.mp4 --fps 15
```

### Patch-Based Processing Pipeline

```bash
# 1. Preprocess
denoise -i raw.nii.gz -o denoised.nii.gz
biascorrect -i denoised.nii.gz -o corrected.nii.gz

# 2. Split into 64³ patches, normalize, skip background
patchifynii -i corrected.nii.gz -o ./patches -p 64 64 64 --norm

# 3. ... process patches (e.g. feed into a model) ...

# 4. Reconstruct
unpatchifynii -i ./patches -o reconstructed.nii.gz
```

### MP2RAGE Processing

```bash
# Process MP2RAGE data with automatic noise estimation
denoiseMP2RAGE --uni MP2RAGE_UNI.nii.gz \
               --inv1 MP2RAGE_INV1.nii.gz \
               --inv2 MP2RAGE_INV2.nii.gz \
               -o MP2RAGE_denoised.nii.gz
```

### TOF MRA Vascular Rendering

```bash
# 1. Denoise the TOF angiogram
denoise -i tof.nii.gz -o tof_denoised.nii.gz

# 2. Static axial MIP for a quick look
mip -i tof_denoised.nii.gz -o tof_mip.nii.gz

# 3. Sliding-slab MIP to scroll through the vasculature
mip -i tof_denoised.nii.gz -o tof_slabmip.nii.gz --slab 20

# 4. Rotating cine for presentation
mip -i tof_denoised.nii.gz -o tof_spin.mp4 --mp4 -a 1 --frames 72 --fps 15
```

### MATLAB Denoising

```bash
# Run the bundled MATLAB Rice/VST denoising routine
mdenoise -i raw.nii.gz -o matlab_denoised.nii.gz
```

## Development

This package is built on top of:
- **TorchIO**: For medical image processing and transformations
- **ANTsPyX**: For advanced image registration
- **NiBabel**: For NIfTI file I/O
- **OpenCV**: For video generation
- **MATLAB**: For the optional MATLAB denoising and SPM12 bias-correction workflows

## Author

**Jinghang Li**  
Email: jinghang.li@pitt.edu

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 0.2.1
- Added `mip`: full, sliding-slab, and rotating cine (GIF/MP4) Maximum Intensity Projections for TOF MRA, with optional CuPy GPU acceleration (`pip install "niiprep[gpu]"`)

### Version 0.2.0
- Added `patchifynii`: split 3D NIfTI volumes into patches with automatic zero-padding, normalization, and Otsu-based background filtering
- Added `unpatchifynii`: reconstruct volumes from patches with automatic crop-back to original shape
- Added `patchify>=0.1.0` dependency
- Added `mdenoise`: MATLAB-based Rice/VST denoising command and Python API

### Version 0.1.0
- Initial release
- Basic resampling, registration, and conversion tools
- MP2RAGE denoising functionality
- Command-line interface for all tools
