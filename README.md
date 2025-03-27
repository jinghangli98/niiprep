# niiprep

A CLI wrapper for TorchIO and ANTsPyX for NIfTI image processing.

## Features

- Resample NIfTI images to specified resolution
- Register NIfTI images using ANTsPyX
- Convert NIfTI images to MP4 videos
- Round NIfTI pixel values
- Denoise MP2RAGE images for improved T1w contrast
- Register images using VoxelMorph (deep learning-based registration) with multiple pre-trained models included

## Installation

```bash
pip install niiprep
```

With VoxelMorph support:

```bash
pip install niiprep[voxelmorph]
```

For development:

```bash
git clone https://github.com/yourusername/niiprep.git
cd niiprep
pip install -e .
```

## Usage

### Resampling

```bash
resample -i input.nii.gz -o output.nii.gz -s 1.0 1.0 1.0 --interpolation linear
```

### Registration

```bash
registernii -f fixed.nii.gz -m moving.nii.gz -o registered.nii.gz -t syn --interpolation linear
```

### Convert NIfTI to MP4

```bash
nii2mp4 -i input.nii.gz -o output.mp4 -d 2 --fps 10
```

### Round NIfTI pixel values

```bash
roundnii -i input.nii.gz
```

### Denoise MP2RAGE images

```bash
denoiseMP2RAGE --uni uni.nii.gz --inv1 inv1.nii.gz --inv2 inv2.nii.gz --output denoised.nii.gz --regularization 1.0
```

### VoxelMorph Registration

Requires optional dependency: `pip install niiprep[voxelmorph]`

```bash
# Using the default pre-trained model (T1_BRAIN)
vxmreg -m moving.nii.gz -f fixed.nii.gz -o moved.nii.gz

# Using a specific predefined model
vxmreg -m moving.nii.gz -f fixed.nii.gz -o moved.nii.gz --model-type BRAINS_DICE

# List all available predefined models
vxmreg --list-models

# Or specify a custom model file path
vxmreg -m moving.nii.gz -f fixed.nii.gz -o moved.nii.gz --model path/to/custom/model.pt --warp warp.nii.gz
```

#### Included VoxelMorph Models

The package includes the following pre-trained VoxelMorph models:

1. **T1_BRAIN**: Default model for T1-weighted brain MRI registration with MSE similarity metric. Small and fast.
2. **BRAINS_DICE**: Specialized brain registration model with Dice similarity and velocity field regularization. Larger but more accurate for brain images.
3. **SHAPES_DICE**: Generic shapes registration with Dice similarity and velocity field. Versatile for different types of images.

> **Note:** The PyTorch version requires models in .pt format. The package will try to create placeholder models automatically. If this fails, you can create them manually with:
> ```bash
> python -m niiprep.convert_models
> ```
> Note that these are placeholder models without trained weights. The registration will still work, but may not be optimal.

## Python API

You can also use niiprep as a Python package:

```python
from niiprep import resample, register, nii_to_mp4

# Resample a NIfTI image
resample(
    input_path="input.nii.gz",
    output_path="resampled.nii.gz",
    target_spacing=(1.0, 1.0, 1.0)
)

# Register a moving image to a fixed image
register(
    fixed_path="fixed.nii.gz",
    moving_path="moving.nii.gz",
    output_path="registered.nii.gz",
    reg_type="syn"
)

# Convert a NIfTI image to MP4
nii_to_mp4(
    input_path="input.nii.gz",
    output_path="output.mp4",
    dimension=2,
    fps=10
)

# Register using VoxelMorph (requires voxelmorph package)
try:
    from niiprep import register_voxelmorph, ModelType
    
    # Using the default pre-trained model (T1_BRAIN)
    register_voxelmorph(
        moving_path="moving.nii.gz",
        fixed_path="fixed.nii.gz",
        moved_path="moved.nii.gz"
    )
    
    # Using a specific predefined model
    register_voxelmorph(
        moving_path="moving.nii.gz",
        fixed_path="fixed.nii.gz",
        moved_path="moved.nii.gz",
        model_type=ModelType.BRAINS_DICE  # Or use the string 'BRAINS_DICE'
    )
    
    # Or specify a custom model
    register_voxelmorph(
        moving_path="moving.nii.gz",
        fixed_path="fixed.nii.gz",
        moved_path="moved.nii.gz",
        model_path="path/to/custom/model.pt",
        warp_path="warp.nii.gz"  # Optional
    )
    
    # List available models
    from niiprep import list_available_models
    available_models = list_available_models()
    for name, desc in available_models.items():
        print(f"{name}: {desc}")
        
except ImportError:
    print("VoxelMorph not installed. Install with: pip install voxelmorph")
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
