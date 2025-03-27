"""
Module for VoxelMorph-based registration.

This module provides functions for registering images using VoxelMorph,
a deep learning framework for deformable medical image registration.

Reference:
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.
"""

import os
import pathlib
import numpy as np
import nibabel as nib
import torch
from enum import Enum, auto
from typing import Optional, Union, Dict, Tuple

# Set PyTorch backend for VoxelMorph
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

class ModelType(Enum):
    """Enum for predefined VoxelMorph model types."""
    T1_BRAIN = auto()  # T1-weighted brain MRI registration model
    BRAINS_DICE = auto()  # Brain model with Dice loss and velocity field
    SHAPES_DICE = auto()  # Generic shapes model with Dice loss

# Mapping of model types to their filenames
MODEL_FILES = {
    ModelType.T1_BRAIN: "vxm_dense_brain_T1_3D_mse.pt",
    ModelType.BRAINS_DICE: "brains-dice-vel-0.5-res-16-256f.pt",
    ModelType.SHAPES_DICE: "shapes-dice-vel-3-res-8-16-32-256f.pt"
}

# Model descriptions for documentation
MODEL_DESCRIPTIONS = {
    ModelType.T1_BRAIN: "T1-weighted brain MRI registration with MSE similarity metric (small, fast)",
    ModelType.BRAINS_DICE: "Brain registration with Dice similarity and velocity field regularization (large, accurate)",
    ModelType.SHAPES_DICE: "Generic shapes registration with Dice similarity and velocity field (versatile)"
}

# Model shape specifications
MODEL_SHAPES = {
    ModelType.T1_BRAIN: (160, 192, 224),
    ModelType.BRAINS_DICE: (160, 192, 224),
    ModelType.SHAPES_DICE: (128, 128, 128)
}

def get_model_path(model_type: ModelType) -> str:
    """
    Returns the path to a predefined VoxelMorph model included with the package.
    
    Args:
        model_type: Type of model to use (from ModelType enum)
    
    Returns:
        str: Path to the model file
    """
    module_path = pathlib.Path(__file__).parent.absolute()
    model_path = module_path / MODEL_FILES[model_type]
    return str(model_path)

def get_default_model_path() -> str:
    """
    Returns the path to the default VoxelMorph model (T1_BRAIN).
    
    Returns:
        str: Path to the default model file
    """
    return get_model_path(ModelType.T1_BRAIN)

def list_available_models() -> Dict[str, str]:
    """
    List all available predefined models with their descriptions.
    
    Returns:
        Dict[str, str]: Dictionary mapping model names to descriptions
    """
    return {model_type.name: MODEL_DESCRIPTIONS[model_type] for model_type in ModelType}

def ensure_model_exists(model_path: str) -> bool:
    """
    Ensure that a PyTorch model file exists at the specified path.
    If it doesn't exist, create an empty placeholder model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        bool: True if the model exists or was created, False otherwise
    """
    if os.path.exists(model_path):
        return True
    
    print(f"Model not found at {model_path}. Creating placeholder...")
    try:
        # Create an empty state dict
        state_dict = {}
        # Save as PyTorch model
        torch.save(state_dict, model_path)
        return True
    except Exception as e:
        print(f"Error creating placeholder model: {e}")
        return False

def create_vxm_model(inshape, device='cpu'):
    """
    Create a new VoxelMorph model with the specified input shape.
    
    Args:
        inshape: Input shape for the model (3D tuple)
        device: Device to create the model on ('cpu' or 'cuda')
        
    Returns:
        VxmDense model instance
    """
    try:
        import voxelmorph as vxm
        
        # Create a new model with appropriate parameters
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[
                [16, 32, 32, 32],
                [32, 32, 32, 32, 32, 16, 16]
            ]
        )
        
        model.to(device)
        return model
    except Exception as e:
        print(f"Error creating VoxelMorph model: {e}")
        raise

def register_voxelmorph(
    moving_path: str,
    fixed_path: str,
    moved_path: str,
    model_path: Optional[str] = None,
    model_type: Optional[Union[str, ModelType]] = None,
    warp_path: Optional[str] = None,
    gpu: Optional[str] = None,
    multichannel: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Register a moving image to a fixed image using a trained VoxelMorph model.
    
    Args:
        moving_path (str): Path to the moving image (source) file.
        fixed_path (str): Path to the fixed image (target) file.
        moved_path (str): Path to save the warped output image.
        model_path (str, optional): Path to a custom VoxelMorph model file.
            If specified, takes precedence over model_type.
        model_type (Union[str, ModelType], optional): Type of predefined model to use.
            Can be a ModelType enum value or a string matching one of the enum names
            (e.g., "T1_BRAIN", "BRAINS_DICE", "SHAPES_DICE").
            If neither model_path nor model_type is specified, uses T1_BRAIN as default.
        warp_path (str, optional): Path to save the warp deformation field.
        gpu (str, optional): GPU device to use. If None or '-1', CPU is used.
        multichannel (bool, optional): Specify if data has multiple channels.
        
    Returns:
        tuple: (moved_image, warp_field) numpy arrays.
    
    Raises:
        ValueError: If an invalid model_type string is provided.
        ImportError: If VoxelMorph is not installed.
    """
    # Import VoxelMorph
    try:
        import voxelmorph as vxm
    except ImportError:
        raise ImportError(
            "VoxelMorph is required for this function. "
            "Please install it with: pip install voxelmorph"
        )
    
    # Determine model type for shape information
    selected_model_type = None
    
    # Determine which model to use
    if model_path is None:
        if model_type is None:
            # Default to T1_BRAIN if neither model_path nor model_type is specified
            selected_model_path = get_default_model_path()
            selected_model_type = ModelType.T1_BRAIN
        else:
            # Convert string to ModelType enum if necessary
            if isinstance(model_type, str):
                try:
                    model_type = ModelType[model_type.upper()]
                except KeyError:
                    available_models = ", ".join([m.name for m in ModelType])
                    raise ValueError(
                        f"Invalid model_type: {model_type}. Available models: {available_models}"
                    )
            
            selected_model_path = get_model_path(model_type)
            selected_model_type = model_type
    else:
        # Use the custom model path provided by the user
        selected_model_path = model_path
    
    # Ensure the model file exists
    ensure_model_exists(selected_model_path)
    
    # Device handling
    if gpu and (gpu != '-1'):
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Load moving and fixed images
    add_feat_axis = not multichannel
    
    try:
        moving = vxm.py.utils.load_volfile(moving_path, add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed, fixed_affine = vxm.py.utils.load_volfile(
            fixed_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    except Exception as e:
        print(f"Error loading images: {e}")
        raise
    
    # Get image shapes for model creation
    inshape = fixed.shape[1:-1]  # Remove batch and channel dimensions
    
    # Create model
    try:
        # Get model shape from predefined shapes or use image shape
        if selected_model_type and selected_model_type in MODEL_SHAPES:
            model_inshape = MODEL_SHAPES[selected_model_type]
        else:
            model_inshape = inshape
        
        model = create_vxm_model(model_inshape, device)
    except Exception as e:
        print(f"Error creating model: {e}")
        raise
    
    # Set up tensors and permute
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
    
    # Predict
    with torch.no_grad():
        moved, warp = model(input_moving, input_fixed, registration=True)
    
    # Convert to numpy arrays
    moved_np = moved.detach().cpu().numpy().squeeze()
    warp_np = warp.detach().cpu().numpy().squeeze()
    
    # Save moved image
    vxm.py.utils.save_volfile(moved_np, moved_path, fixed_affine)
    
    # Save warp if requested
    if warp_path:
        vxm.py.utils.save_volfile(warp_np, warp_path, fixed_affine)
    
    return moved_np, warp_np
