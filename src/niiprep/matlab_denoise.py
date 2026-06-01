import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib


def _lavi_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "denoise_lavi"


def _matlab_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_matlab(matlab_cmd: str, cores: int = None) -> None:
    """Run a MATLAB command in batch mode with a sanitized LD_LIBRARY_PATH."""
    env = os.environ.copy()
    # Strip any user-local GCC lib paths that ship an older libstdc++ than
    # MATLAB R2025a requires (needs GLIBCXX_3.4.29+, CXXABI_1.3.13+). These
    # commonly leak in via LD_LIBRARY_PATH and/or an LD_PRELOAD of
    # libstdc++.so.6 (e.g. ~/opt/gcc-*/lib64) set in the user's shell rc.
    def _scrub(value: str) -> str:
        return ":".join(p for p in value.split(":") if "opt/gcc" not in p and p)

    env["LD_LIBRARY_PATH"] = _scrub(env.get("LD_LIBRARY_PATH", ""))
    env["LD_PRELOAD"] = _scrub(env.get("LD_PRELOAD", ""))

    # Cap the OpenMP-threaded BM4D mex at the requested core count.
    if cores is not None:
        env["OMP_NUM_THREADS"] = str(cores)

    subprocess.run(
        ["matlab", "-nodisplay", "-nosplash", "-r", matlab_cmd],
        check=True,
        env=env,
    )


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
            f"I_noisy = double(niftiread({_matlab_string(matlab_input)})); "
            "Sigma_map = rice_sigma_mapEST(I_noisy); "
            f"I_denoised = {denoise_call}; "
            f"niftiwrite(cast(I_denoised, info.Datatype), {_matlab_string(matlab_output)}, info); "
            "exit"
        )

        _run_matlab(matlab_cmd, cores=cores)

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
