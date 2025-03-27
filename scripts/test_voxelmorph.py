#!/usr/bin/env python3
"""
Test script for VoxelMorph registration in niiprep.
This script tests the VoxelMorph registration without actually running registration.
"""

import os
import sys
from pathlib import Path

def test_voxelmorph_models():
    """Test the VoxelMorph model loading functionality."""
    try:
        # Import required modules
        from niiprep import ModelType, get_model_path, list_available_models
        
        # List available models
        print("Available models:")
        for name, desc in list_available_models().items():
            print(f"  {name}: {desc}")
        
        # Get model paths
        for model_type in ModelType:
            model_path = get_model_path(model_type)
            print(f"Model path for {model_type.name}: {model_path}")
            
            # Check if the model file exists
            if os.path.exists(model_path):
                print(f"  Model file exists: {os.path.getsize(model_path)} bytes")
            else:
                print(f"  Model file does not exist")
                
        return True
    except Exception as e:
        print(f"Error testing VoxelMorph: {e}")
        return False

def main():
    """Run the test script."""
    print("Testing VoxelMorph functionality in niiprep...")
    
    success = test_voxelmorph_models()
    
    if success:
        print("\nTest completed successfully.")
        return 0
    else:
        print("\nTest failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
