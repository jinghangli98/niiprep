import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib


def _denoise_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "denoise"


def _matlab_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def mdenoise(input_path: str, output_path: str) -> None:
    """Run the MATLAB denoising routine and save its first output."""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    assets_dir = _denoise_assets_dir()
    if not (assets_dir / "new_denoising_func_matlab_matrix.m").exists():
        raise FileNotFoundError(f"MATLAB denoise assets were not found in {assets_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_img = nib.load(input_path)
        matlab_input = os.path.join(tmpdir, "input.nii")
        matlab_output = os.path.join(tmpdir, "d1.nii")
        nib.save(input_img, matlab_input)

        matlab_cmd = (
            f"addpath(genpath({_matlab_string(str(assets_dir))})); "
            f"img = niftiread({_matlab_string(matlab_input)}); "
            f"info = niftiinfo({_matlab_string(matlab_input)}); "
            "[d1, ~] = new_denoising_func_matlab_matrix(img, class(img)); "
            f"niftwrite(d1, {_matlab_string(matlab_output)}, info); "
            "exit"
        )

        subprocess.run(
            ["matlab", "-nodisplay", "-nosplash", "-r", matlab_cmd],
            check=True,
        )

        if not os.path.exists(matlab_output):
            raise RuntimeError(f"MATLAB denoise output not found at expected path: {matlab_output}")

        denoised_img = nib.load(matlab_output)
        nib.save(denoised_img, output_path)
        print(f"MATLAB-denoised image saved to: {output_path}")
