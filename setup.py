from setuptools import setup, find_packages

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="niiprep",
    version="0.1.0",
    author="Jinghang Li",
    author_email="jinghang.li@pitt.edu",
    description="A CLI wrapper for TorchIO and ANTsPyX for NIfTI image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/niiprep",  # Replace with your repository URL
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/niiprep/issues",  # Replace with your issue tracker
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        'niiprep': ['*.h5', '*.pt'],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torchio>=0.18.0",
        "antspyx>=0.3.0",
        "nibabel>=3.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "voxelmorph": ["voxelmorph>=0.2", "torch>=1.9.0"],
    },
    
    entry_points={
        'console_scripts': [
            'resample=niiprep.cli:resample_cli',
            'registernii=niiprep.cli:register_cli',
            'nii2mp4=niiprep.cli:nii_to_mp4_cli',
            'roundnii=niiprep.cli:round_cli',
            'denoiseMP2RAGE=niiprep.cli:denoise_mp2rage',
            'vxmreg=niiprep.cli:voxelmorph_register_cli'
        ],
    },
)
