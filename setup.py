from setuptools import setup, find_packages


setup(
    name="niiprep",
    version="0.1.0",
    author="Jinghang Li",
    author_email="jinghang.li@pitt.edu",
    description="A CLI wrapper for TorchIO and ANTsPyX for NIfTI image processing",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        'niiprep': [''],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torchio>=0.18.0",
        "antspyx>=0.3.0",
        "nibabel>=3.0.0",
        "numpy>=1.19.0",
    ],
    entry_points={
        'console_scripts': [
            'resample=niiprep.cli:resample_cli',
            'registernii=niiprep.cli:register_cli',
            'nii2mp4=niiprep.cli:nii_to_mp4_cli',
            'roundnii=niiprep.cli:round_cli',
            'denoiseMP2RAGE=niiprep.cli:denoise_mp2rage',
            'crop=niiprep.cli:crop_cli'
        ],
    },
) 