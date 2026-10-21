import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform, binary_fill_holes, label

from .synthstrip_helper import SynthStripHelper, extend_sdt

# SynthStrip was trained on 1 mm isotropic volumes in LIA orientation:
# voxel axes i, j, k point to world -x (Left), -z (Inferior), +y (Anterior).
_LIA = np.array([[-1, 0, 0],
                 [0, 0, 1],
                 [0, -1, 0]], dtype=float)


def _conform_affine(data, affine):
    """Return (shape, affine) of a 1 mm LIA grid centered on the head.

    Mirrors SynthStrip's conform + crop_to_bbox + reshape: the grid is
    centered on the bounding box of nonzero voxels and sized to a
    multiple of 64 between 192 and 320 per axis.
    """
    nz = np.argwhere(data > 0)
    if nz.size == 0:
        nz = np.argwhere(np.ones(data.shape, bool))
    lo, hi = nz.min(0), nz.max(0)
    corners = np.array([[x, y, z, 1.0]
                        for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    world = (affine @ corners.T)[:3].T
    # extent of the bbox measured along each LIA voxel axis
    proj = world @ _LIA
    extent = proj.max(0) - proj.min(0) + 1
    shape = np.clip(np.ceil(extent / 64).astype(int) * 64, 192, 320)

    center = world.mean(0)
    out = np.eye(4)
    out[:3, :3] = _LIA
    out[:3, 3] = center - _LIA @ ((shape - 1) / 2)
    return tuple(shape), out


def _resample(data, src_affine, dst_shape, dst_affine, order, cval):
    """Resample ``data`` from ``src_affine`` onto the destination grid."""
    m = np.linalg.inv(src_affine) @ dst_affine
    return affine_transform(data, m[:3, :3], m[:3, 3],
                            output_shape=dst_shape, order=order, cval=cval)


def _strip_volume(helper, data, affine, border):
    """Return the uint8 brain mask of a single 3D volume."""
    shape, conf_affine = _conform_affine(data, affine)
    conf = _resample(data, affine, shape, conf_affine, order=1,
                     cval=float(data.min())).astype(np.float32)

    conf -= conf.min()
    p99 = np.percentile(conf, 99)
    if p99 > 0:
        conf = np.clip(conf / p99, 0, 1)

    import torch
    inp = torch.from_numpy(conf[None, None]).to(helper.device)
    with torch.no_grad():
        sdt = helper.model(inp).squeeze().cpu().numpy()

    sdt = extend_sdt(sdt, border=border)
    # Bring the distance map back to the input grid; outside the
    # conformed FOV counts as far from the brain.
    sdt = _resample(sdt, conf_affine, data.shape, affine, order=1,
                    cval=100.0)

    mask = sdt < border
    labeled, n = label(mask)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        mask = labeled == sizes.argmax()
    return binary_fill_holes(mask).astype(np.uint8)


def skullstrip(input_path: str, output_path: str = None,
               mask_path: str = None, border: float = 1.0,
               fill: float = None, model_path: str = None,
               use_gpu: bool = True, no_csf: bool = False) -> None:
    """
    Skull-strip a brain MRI with SynthStrip (any contrast or resolution).

    The image is conformed to the 1 mm LIA grid the network expects,
    and the predicted brain boundary is resampled back, so outputs are
    on the input's original grid.

    Args:
        input_path: Path to input NIfTI file (3D, or 4D stripped per frame).
        output_path: Path to save the skull-stripped image.
        mask_path: Path to save the binary brain mask.
        border: Mask border threshold in mm (default 1).
        fill: Background fill value; defaults to min(image.min(), 0).
        model_path: Alternative SynthStrip weights. Defaults to the model
            bundled with niiprep.
        use_gpu: Use CUDA when available; falls back to the CPU otherwise.
        no_csf: Exclude CSF from the brain border (needs FreeSurfer's
            synthstrip.nocsf.1.pt via FREESURFER_HOME or model_path).
    """
    if output_path is None and mask_path is None:
        raise ValueError("provide at least one of output_path or mask_path")

    import torch
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"SynthStrip device: {device}")
    helper = SynthStripHelper(model_path=model_path, device=device,
                              no_csf=no_csf)

    img = nib.load(input_path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 3:
        mask = _strip_volume(helper, data, img.affine, border)
    elif data.ndim == 4:
        mask = np.stack([_strip_volume(helper, data[..., f], img.affine,
                                       border)
                         for f in range(data.shape[3])], axis=-1)
    else:
        raise ValueError(f"Expected a 3D or 4D image, got shape {data.shape}")

    if mask_path:
        hdr = img.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(mask, img.affine, hdr), mask_path)
        print(f"Brain mask saved to {mask_path}")

    if output_path:
        if fill is None:
            fill = min(float(data.min()), 0.0)
        brain = np.asanyarray(img.dataobj).copy()
        brain[mask == 0] = fill
        nib.save(nib.Nifti1Image(brain, img.affine, img.header), output_path)
        print(f"Skull-stripped image saved to {output_path}")
