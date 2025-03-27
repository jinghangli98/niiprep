from .resample import resample
from .registration import register
from .nii2mp4 import nii_to_mp4
from .round import round_nifti
from .denoise_mp2rage import robust_combination

# Import VoxelMorph registration if available
try:
    import voxelmorph
    from .voxelmorph_reg import (
        register_voxelmorph, 
        list_available_models, 
        ModelType,
        get_model_path
    )
    _has_voxelmorph = True
except ImportError:
    _has_voxelmorph = False

__version__ = "0.1.0"
__all__ = ["resample", "register", "nii_to_mp4", "round_nifti", "robust_combination"]

if _has_voxelmorph:
    __all__.extend(["register_voxelmorph", "list_available_models", "ModelType", "get_model_path"])
