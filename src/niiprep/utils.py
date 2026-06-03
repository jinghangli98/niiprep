"""DICOM I/O helpers built on SimpleITK + pydicom.

Ported from the nexus_imaging utilities, trimmed to the two functions needed
for MP2RAGE DICOM support (no torch/torchio dependencies).
"""

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pydicom


def load_dicom_with_sitk(folder_path, reorient=False):
    """Load a DICOM series using SimpleITK.

    Returns ``(sitk_image, array, affine)`` where ``array`` is in (x, y, z)
    order and ``affine`` is a 4x4 RAS affine.
    """
    if os.path.isfile(folder_path):
        folder_path = os.path.dirname(folder_path) or '.'

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder_path)
    if not series_ids:
        raise ValueError("No DICOM series found.")

    series_file_names = reader.GetGDCMSeriesFileNames(folder_path, series_ids[0])
    reader.SetFileNames(series_file_names)

    image = reader.Execute()

    if reorient:
        image = sitk.DICOMOrient(image, 'RAS')

    array = sitk.GetArrayFromImage(image)
    array = array.transpose(2, 1, 0)  # (z,y,x) -> (x,y,z)

    # Build affine in LPS (SimpleITK native)
    affine = np.eye(4)
    affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) * np.array(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()

    # Convert LPS -> RAS: negate first two rows
    affine[0, :] *= -1
    affine[1, :] *= -1

    return image, array, affine


def save_enhanced_dicom(enhanced_array, original_dicom_folder, output_folder,
                        new_spacing=(0.55, 0.55, 0.55), series_description_suffix="_SuperRes"):
    """Save an array as a DICOM series, preserving the original metadata.

    Args:
        enhanced_array: image array (x, y, z) in RAS order.
        original_dicom_folder: path to the original DICOM series (metadata template).
        output_folder: directory to write the new DICOM files into.
        new_spacing: spacing of the output image (x, y, z).
        series_description_suffix: suffix appended to the SeriesDescription.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # --- Read original DICOM in both native and RAS orientations ---
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(original_dicom_folder)
    if not series_ids:
        raise ValueError("No DICOM series found.")
    series_file_names = reader.GetGDCMSeriesFileNames(original_dicom_folder, series_ids[0])

    # Native (non-reoriented) image for reference geometry
    reader.SetFileNames(series_file_names)
    native_image = reader.Execute()

    # RAS-reoriented image to get RAS geometry (direction, origin in LPS internally)
    ras_image = sitk.DICOMOrient(native_image, 'RAS')

    # --- Convert enhanced RAS array back to native DICOM space ---
    # (x, y, z) -> (z, y, x) for SimpleITK
    enhanced_sitk_array = enhanced_array.transpose(2, 1, 0)
    enhanced_image = sitk.GetImageFromArray(enhanced_sitk_array)

    # Set RAS geometry on the enhanced image (SimpleITK stores in LPS internally)
    enhanced_image.SetDirection(ras_image.GetDirection())
    enhanced_image.SetOrigin(ras_image.GetOrigin())
    enhanced_image.SetSpacing([float(s) for s in new_spacing])

    # Get original orientation code and reorient back to native
    orig_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        native_image.GetDirection()
    )
    print(f"Original DICOM orientation: {orig_orient}, reorienting from RAS back to {orig_orient}")
    native_enhanced = sitk.DICOMOrient(enhanced_image, orig_orient)

    # Extract array in native DICOM order (z, y, x) — matches original metadata
    native_array = sitk.GetArrayFromImage(native_enhanced)
    native_spacing = native_enhanced.GetSpacing()
    native_origin = native_enhanced.GetOrigin()
    native_direction = np.array(native_enhanced.GetDirection()).reshape(3, 3)

    num_slices = native_array.shape[0]
    print(f"Saving {num_slices} slices, array shape: {native_array.shape}")

    # --- Save each slice ---
    new_series_uid = pydicom.uid.generate_uid()

    for i in range(num_slices):
        # Slice position in LPS (step along slice normal = direction column 2)
        slice_position = (np.array(native_origin) +
                          i * native_spacing[2] * native_direction[:, 2])

        # Copy metadata from nearest original slice
        ds = pydicom.dcmread(series_file_names[min(i, len(series_file_names) - 1)])

        # Update pixel data
        slice_data = native_array[i]
        ds.PixelData = slice_data.astype(np.int16).tobytes()

        # Update geometry tags
        ds.Rows, ds.Columns = slice_data.shape
        ds.PixelSpacing = [float(native_spacing[1]), float(native_spacing[0])]

        if hasattr(ds, 'SliceThickness'):
            ds.SliceThickness = float(native_spacing[2])
        if hasattr(ds, 'SpacingBetweenSlices'):
            ds.SpacingBetweenSlices = float(native_spacing[2])

        ds.ImagePositionPatient = [float(slice_position[0]),
                                   float(slice_position[1]),
                                   float(slice_position[2])]

        ds.SeriesInstanceUID = new_series_uid
        if hasattr(ds, 'SeriesDescription'):
            ds.SeriesDescription = str(ds.SeriesDescription) + series_description_suffix
        else:
            ds.SeriesDescription = "MP2RAGE"

        ds.InstanceNumber = i + 1
        ds.SOPInstanceUID = pydicom.uid.generate_uid()

        output_file = os.path.join(output_folder, f"enhanced_{i:04d}.dcm")
        ds.save_as(output_file)

    print(f"Saved {num_slices} DICOM files to {output_folder}")
