### Example of usage

```
img = niftiread(nifti_file_path);
info = niftiinfo(nifti_file_path);

[ d1, d2 ] = new_denoising_func_matlab_matrix(img);

niftwrite(d1, 'd1.nii', info, 'Compressed', true)
niftwrite(d2, 'd2.nii', info, 'Compressed', true)
```
