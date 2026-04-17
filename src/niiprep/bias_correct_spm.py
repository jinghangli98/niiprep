import os
import shutil
import subprocess
import tempfile
import nibabel as nib
from .round import round_nifti

DEFAULT_SPM_PATH = '/ihome/tibrahim/jil202/spm12'


def _get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def spm_bias_correct(
    input_path,
    output_path,
    spm_path=DEFAULT_SPM_PATH,
    window_size=60,
    sphere_size=50,
    include_c3=False,
    save_masks=False,
    round_output=False,
):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        if input_path.endswith('.nii.gz'):
            img = nib.load(input_path)
            tmp_nii = os.path.join(tmpdir, 'input.nii')
            nib.save(img, tmp_nii)
            nii_input = tmp_nii
        elif input_path.endswith('.nii'):
            nii_input = os.path.join(tmpdir, os.path.basename(input_path))
            shutil.copy2(input_path, nii_input)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")

        basename = os.path.splitext(os.path.basename(nii_input))[0]
        script_dir = _get_script_dir()
        include_c3_val = '1' if include_c3 else '0'

        matlab_cmd = (
            f"addpath('{script_dir}'); "
            f"bias_correct('{nii_input}', '{spm_path}', {window_size}, {sphere_size}, {include_c3_val}); "
            f"exit"
        )

        subprocess.run(
            ['matlab', '-nodisplay', '-nosplash', '-r', matlab_cmd],
            check=True,
        )

        spm_output = os.path.join(tmpdir, f'm{basename}.nii')
        if not os.path.exists(spm_output):
            raise RuntimeError(f"SPM output not found at expected path: {spm_output}")

        corrected_img = nib.load(spm_output)
        nib.save(corrected_img, output_path)
        print(f"Bias-corrected image saved to: {output_path}")
        if round_output:
            round_nifti(output_path)

        if save_masks:
            out_dir = os.path.dirname(output_path)
            out_stem = os.path.basename(output_path)
            for ext in ('.nii.gz', '.nii'):
                if out_stem.endswith(ext):
                    out_stem = out_stem[: -len(ext)]
                    break

            for prefix in ['c1', 'c2', 'c3', 'c4', 'c5']:
                mask_file = os.path.join(tmpdir, f'{prefix}{basename}.nii')
                if os.path.exists(mask_file):
                    dest = os.path.join(out_dir, f'{prefix}_{out_stem}.nii')
                    shutil.copy2(mask_file, dest)
                    print(f"Saved mask: {dest}")

