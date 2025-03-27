#!/usr/bin/env python3
"""
Script to create placeholder PyTorch models without requiring TensorFlow or VoxelMorph.
Run this script after installing the package to create PyTorch models.

Example:
    python -m niiprep.convert_models
"""

import os
import sys
import pathlib

def create_empty_pytorch_models():
    """Create empty PyTorch models without requiring TensorFlow or VoxelMorph."""
    try:
        import torch
        
        # Get the module directory
        module_path = pathlib.Path(__file__).parent.absolute()
        
        # List of model names
        model_names = [
            "vxm_dense_brain_T1_3D_mse",
            "brains-dice-vel-0.5-res-16-256f",
            "shapes-dice-vel-3-res-8-16-32-256f"
        ]
        
        success_count = 0
        
        # For each model
        for model_name in model_names:
            pt_path = module_path / f"{model_name}.pt"
            
            # Skip if PyTorch model already exists
            if pt_path.exists():
                print(f"PyTorch model already exists: {pt_path}")
                success_count += 1
                continue
            
            print(f"Creating PyTorch placeholder model: {pt_path}")
            
            # Create an empty state dict (dictionary of tensors)
            # This is a very simplified placeholder, not a real model
            state_dict = {}
            
            # Save the state dict as a PyTorch model file
            torch.save(state_dict, pt_path)
            print(f"Created PyTorch placeholder model: {pt_path}")
            success_count += 1
        
        if success_count == len(model_names):
            print("All placeholder models created successfully.")
            return True
        else:
            print(f"Created {success_count} of {len(model_names)} models.")
            return success_count > 0
    
    except Exception as e:
        print(f"Error creating models: {e}")
        return False

if __name__ == "__main__":
    print("Creating placeholder PyTorch models...")
    success = create_empty_pytorch_models()
    if success:
        print("Conversion completed successfully.")
        sys.exit(0)
    else:
        print("Conversion failed.")
        sys.exit(1)
