# NiiPrep

A CLI wrapper for TorchIO and ANTsPyX for NIfTI image processing

## Overview

NiiPrep is a Python package that provides convenient command-line tools for common neuroimaging preprocessing tasks. It combines the power of TorchIO and ANTsPyX libraries to offer streamlined workflows for NIfTI image manipulation, registration, resampling, and visualization.

## Features

- **Image Resampling**: Change voxel spacing with multiple interpolation methods
- **Image Registration**: Rigid, affine, and deformable registration using ANTsPyX
- **Video Conversion**: Convert NIfTI files to MP4 videos for visualization
- **Image Cropping/Padding**: Resize images to specified dimensions or automatically remove empty space
- **Value Rounding**: Round pixel values in NIfTI images
- **MP2RAGE Denoising**: Robust combination processing for MP2RAGE images

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
```

**Parameters:**
- `-f, --fixed`: Path to fixed/reference NIfTI file
- `-m, --moving`: Path to moving NIfTI file
- `-o, --output`: Path to save registered NIfTI file
- `-t, --type`: Registration type (rigid, affine, syn), default: syn
- `--interpolation`: Interpolation type, default: linear

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

### 6. `denoiseMP2RAGE` - MP2RAGE Denoising

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

### 7. `autocrop` - Automatic Image Cropping

Automatically crop NIfTI images to remove empty space (based on gradient detection) and optionally pad to a specific shape or multiple.

```bash
autocrop -i input.nii.gz -o output.nii.gz -n 14
```

**Parameters:**
- `-i, --input`: Path to input NIfTI file
- `-o, --output`: Path to save cropped/padded NIfTI file
- `-n`: Optional integer. If provided, output dimensions will be padded to be multiples of this value. Default is None (tight crop).
- `-s, --shape`: Optional target shape (x y z). If provided, output will maximize this shape. If used with `-n`, dimensions are adjusted to the closest smaller multiple of `n`.

## Python API

You can also use NiiPrep functions directly in Python:

```python
from niiprep import resample, register, nii_to_mp4

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
```

## Examples

### Basic Preprocessing Pipeline

```bash
# 1. Resample to 1mm isotropic
resample -i raw.nii.gz -o resampled.nii.gz -s 1.0 1.0 1.0

# 2. Crop to standard size (or use autocrop for automatic sizing)
crop -i resampled.nii.gz -o cropped.nii.gz -s 256 256 256
# OR
autocrop -i resampled.nii.gz -o cropped.nii.gz -n 14

# 3. Register to template
registernii -f template.nii.gz -m cropped.nii.gz -o registered.nii.gz -t affine

# 4. Create visualization video
nii2mp4 -i registered.nii.gz -o preview.mp4 --fps 15
```

### MP2RAGE Processing

```bash
# Process MP2RAGE data with automatic noise estimation
denoiseMP2RAGE --uni MP2RAGE_UNI.nii.gz \
               --inv1 MP2RAGE_INV1.nii.gz \
               --inv2 MP2RAGE_INV2.nii.gz \
               -o MP2RAGE_denoised.nii.gz
```

## Development

This package is built on top of:
- **TorchIO**: For medical image processing and transformations
- **ANTsPyX**: For advanced image registration
- **NiBabel**: For NIfTI file I/O
- **OpenCV**: For video generation

## Author

**Jinghang Li**  
Email: jinghang.li@pitt.edu

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 0.1.0
- Initial release
- Basic resampling, registration, and conversion tools
- MP2RAGE denoising functionality
- Command-line interface for all tools