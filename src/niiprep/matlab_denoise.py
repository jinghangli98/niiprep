import os
import tempfile
from pathlib import Path

import nibabel as nib

from .matlab_runner import run_matlab


def _lavi_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "denoise_lavi"


def _matlab_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def mdenoise(input_path: str, output_path: str, profile: str = "np",
             cores: int = None) -> None:
    """Denoise a structural NIfTI image using the LAVI VST + BM4D pipeline.

    Estimates a non-stationary Rician noise map and applies a
    variance-stabilizing transform followed by BM4D denoising
    (Campos et al.).

    Parameters
    ----------
    input_path, output_path : str
        Paths to the input and output NIfTI files.
    profile : str
        BM4D aggressiveness: ``'lc'`` (less aggressive, fastest),
        ``'np'`` (normal, default), or ``'mp'`` (more aggressive, slowest).
    cores : int, optional
        Number of CPU cores/threads to use. ``None`` (default) lets MATLAB and
        the BM4D mex use all available cores automatically.
    """
    valid_profiles = {"lc", "np", "mp"}
    if profile not in valid_profiles:
        raise ValueError(
            f"Invalid profile {profile!r}; choose one of {sorted(valid_profiles)}"
        )

    if cores is not None and cores < 1:
        raise ValueError(f"cores must be a positive integer, got {cores!r}")

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    assets_dir = _lavi_assets_dir()
    if not (assets_dir / "denoise_with_vst_map.m").exists():
        raise FileNotFoundError(f"LAVI denoise assets were not found in {assets_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_img = nib.load(input_path)
        matlab_input = os.path.join(tmpdir, "input.nii")
        matlab_output = os.path.join(tmpdir, "d1.nii")
        nib.save(input_img, matlab_input)

        # Limit MATLAB's intrinsic multithreading; pass the thread count through
        # to BM4D so the mex honors it too (0/omitted = automatic/all cores).
        thread_prefix = f"maxNumCompThreads({cores}); " if cores is not None else ""
        denoise_call = (
            f"denoise_with_vst_map(I_noisy, Sigma_map, {_matlab_string(profile)}, 1, {cores})"
            if cores is not None
            else f"denoise_with_vst_map(I_noisy, Sigma_map, {_matlab_string(profile)})"
        )

        matlab_cmd = (
            f"addpath(genpath({_matlab_string(str(assets_dir))})); "
            f"{thread_prefix}"
            f"info = niftiinfo({_matlab_string(matlab_input)}); "
            f"I_orig = double(niftiread({_matlab_string(matlab_input)})); "
            # Normalize intensities to 0-255, then scale by 100 before denoising.
            "I_min = min(I_orig(:)); I_max = max(I_orig(:)); "
            "I_range = I_max - I_min; if I_range == 0, I_range = 1; end; "
            "I_scale = 255 * 100 / I_range; "
            "I_noisy = (I_orig - I_min) * I_scale; "
            "Sigma_map = rice_sigma_mapEST(I_noisy); "
            f"I_denoised = {denoise_call}; "
            # Invert the scaling so the output is back in the original intensity range.
            "I_denoised = I_denoised / I_scale + I_min; "
            f"niftiwrite(cast(I_denoised, info.Datatype), {_matlab_string(matlab_output)}, info); "
            "exit"
        )

        run_matlab(matlab_cmd, cores=cores)

        # niftiwrite appends the extension; resolve whichever was produced.
        produced = matlab_output
        if not os.path.exists(produced):
            for cand in (matlab_output + ".nii", matlab_output + ".gz"):
                if os.path.exists(cand):
                    produced = cand
                    break
        if not os.path.exists(produced):
            raise RuntimeError(
                f"LAVI denoise output not found at expected path: {matlab_output}"
            )

        denoised_img = nib.load(produced)
        nib.save(denoised_img, output_path)
        print(f"LAVI-denoised image saved to: {output_path}")
