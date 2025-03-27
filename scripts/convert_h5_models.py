#!/usr/bin/env python3
"""
Script to create placeholder PyTorch models from H5 models.
This script should be run once after installing the package to create PyTorch models.
"""

import os
import sys
import pathlib

def create_empty_pytorch_models():
    """Create empty PyTorch models for all H5 models in the package."""
    try:
        import torch
        import voxelmorph as vxm
        
        # Set backend to PyTorch
        os.environ['NEURITE_BACKEND'] = 'pytorch'
        os.environ['VXM_BACKEND'] = 'pytorch'
        
        # Get the module directory
        try:
            import niiprep
            niiprep_path = pathlib.Path(niiprep.__file__).parent.absolute()
        except ImportError:
            # If running from source
            niiprep_path = pathlib.Path(__file__).parent.parent / "src" / "niiprep"
        
        # List of model names
        model_names = [
            "vxm_dense_brain_T1_3D_mse",
            "brains-dice-vel-0.5-res-16-256f",
            "shapes-dice-vel-3-res-8-16-32-256f"
        ]
        
        # Common brain MRI dimensions - adjust for different models if needed
        model_shapes = {
            "vxm_dense_brain_T1_3D_mse": (160, 192, 224),
            "brains-dice-vel-0.5-res-16-256f": (160, 192, 224),
            "shapes-dice-vel-3-res-8-16-32-256f": (128, 128, 128)
        }
        
        # For each model
        for model_name in model_names:
            h5_path = niiprep_path / f"{model_name}.h5"
            pt_path = niiprep_path / f"{model_name}.pt"
            
            # Skip if PyTorch model already exists
            if pt_path.exists():
                print(f"PyTorch model already exists: {pt_path}")
                continue
            
            print(f"Creating PyTorch model: {pt_path}")
            
            # Create PyTorch model with appropriate shape
            inshape = model_shapes.get(model_name, (160, 192, 224))
            
            # Create model with typical parameters
            model = vxm.networks.VxmDense(
                inshape=inshape,
                nb_unet_features=[
                    [16, 32, 32, 32],
                    [32, 32, 32, 32, 32, 16, 16]
                ]
            )
            
            # Save the model
            torch.save(model.state_dict(), pt_path)
            print(f"Created PyTorch model: {pt_path}")
        
        print("All models converted successfully.")
        return True
    
    except Exception as e:
        print(f"Error converting models: {e}")
        return False

if __name__ == "__main__":
    success = create_empty_pytorch_models()
    if success:
        print("Conversion completed successfully.")
        sys.exit(0)
    else:
        print("Conversion failed.")
        sys.exit(1)
